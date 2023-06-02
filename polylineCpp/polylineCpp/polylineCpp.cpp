#define PY_SSIZE_T_CLEAN
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy\\arrayobject.h"

#pragma warning(disable:4996)		//unsafe functions

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace std;
struct Device {

    queue<void (Device::*) (PyObject* o)> foosQueue;   //очередь функций объекта
    queue<PyObject*> argsQueue;   //очередь аргументов функций

    thread thr; //поток, который будет отвечать за обработку объекта

    condition_variable cvCall; //соответствующие cv
    condition_variable cvProcess;

    mutex mxCall;   //мьютекс по вызовам и добавлению функций в очередь
    mutex mxProcess;    //мьютекс по контролю завершения расчетов

    bool process = true;    //флаг завершения расчетов

    PyArrayObject* pyPolyline;  //указатели на нужные объекты питона
    PyArrayObject* pyCheckList;

    PyArrayObject* pyMaxNlines;
    PyArrayObject* pyNlines;

    double* pPoly_x;
    double* pPoly_y;
    double* pCheckList_i;
    double* pCheckList_t;
    double* pCheckList_isGap;

    double par0;

    size_t* pMaxNlines = nullptr;
    size_t* pNlines = nullptr;

    size_t id = -1;

    Device() = default;

    Device(PyArrayObject* const pyPolyline, PyArrayObject* const pyCheckList, PyArrayObject* const pyMaxNlines, PyArrayObject* const pyNlines,
        PyObject* pyParams, size_t id) :
        par0(0.0)
    {

        if (!setParams(pyParams)) {
            throw new logic_error("");
        }

        //связывание с объектами Py

        pPoly_x = (double*)PyArray_DATA(pyPolyline);   //0-й ряд 
        pPoly_y = pPoly_x + *pMaxNlines;                    //1-й ряд

        pCheckList_i = (double*)PyArray_DATA(pyCheckList);
        pCheckList_t = pCheckList_i + *pMaxNlines;
        pCheckList_isGap = pCheckList_i + *pMaxNlines * 2;

        thr = thread([this]     //запускаем поток
            {
                while (true) {
                    unique_lock lk0(mxCall);
                    cvCall.wait(lk0, [this] {return foosQueue.size(); });   //ждун вызова через pyFun

                    auto foo = foosQueue.front(); //берем функцию с края очереди
                    auto args = argsQueue.front(); //берем аргументы с края очереди

                    if (foo != nullptr && args != nullptr)
                        (this->*foo) (args);    //вызываем функцию с соотв. аргументом

                    foosQueue.pop();  //удаляем из очереди функцию
                    argsQueue.pop();  //удаляем из очереди аргументы

                    if (foosQueue.empty()) {
                        unique_lock lk1(mxProcess);
                        process = true;   //меняем состояние для cv процесса, функция pyFun<nullptr>, т.е. synchronize() из питона, сможет завершиться
                        lk1.unlock();
                        cvProcess.notify_one(); //уведомляем cv о завершении расчетов
                    }

                    lk0.unlock();
                }
            }
        );
    }

    ~Device() {
        //НЕ ОСОБО НУЖНОЕ УДОВОЛЬСТВИЕ, НО ЕСТЬ
        //Питона крашит модули весьма кусочно и не консистентно, поэтому может даже подвесить какую-нибудь работающую функцию.
        //Нам не сильно это важно, так как предполагается существование объекта все время работы программы, ну, или почти все.
        //Тем не менее, если разрабатывать quit() этот деструктор пригодится.

        cout << "Device " << id << " destructor starts" << endl;

        unique_lock lk0(mxProcess);
        cvProcess.wait(lk0, [this] { return process; }); //ждет завершения расчетной очереди функций
        lk0.unlock();

        thr.join(); //оказывается, компилятор и сам завершит поток в бесконечном цикле при разрушении cv

        cout << "Device " << id << " destructor ends normally" << endl;
    }

    bool setParams(PyObject* pyParams) {

        //Можно и getParams() сделать, но пока это необязательно

        if (!PyTuple_Check(pyParams)) {
            cerr << "Parameters should be in a tuple" << endl;
            return false;
        }

        if (PyTuple_GET_SIZE(pyParams) != 1) {  //кол-во параметров, не забываем менять
            cerr << "There should be 1 parameters in a tuple" << endl; // 5-й - кортеж координат монтажного положения устройства
            return false;
        }

        par0 = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 0));

        if (PyErr_Occurred()) {
            cerr << "Wrong type inside the parameters" << endl;
            PyErr_Clear();
            return false;
        }

        return true;
    }

    void closest_pnt(PyObject* o) { }
    void closest_pnt00(PyObject* o) { }
    void check_segment_intersections(PyObject* o) { }
    void check_segment_intersections00(PyObject* o) { }
    void check_if_obb_intersection(PyObject* o) { }

};

//Основной глобальный контейнер устройств (указателей на них)
vector<unique_ptr<Device>> devices; //с unique_ptr будет вывзываться деструктор по завершению отдельного устройства

template<void (Device::* F) (PyObject* o)>
PyObject* pyFun(PyObject*, PyObject* o) {

    if (PyTuple_Check(o)) {
        int id = (int)PyLong_AsLong(PyTuple_GetItem(o, 0));
        if (id >= 0 && id < devices.size()) {

            Device* dev = devices[id].get();

            if (F != nullptr) {
                unique_lock lk(dev->mxCall);
                dev->process = false;   //блочим флаг завершенных расчетов
                dev->foosQueue.push(F);     //добавляем функцию в очередь
                dev->argsQueue.push(o);     //добавляем аргументы в очередь
                lk.unlock();
                dev->cvCall.notify_one();   //уведомляем, проверочная функция в wait запрашивает foosQueue.size()
            }
            else {
                unique_lock lk(dev->mxProcess); //синхронизация, здесь он ждет завершения расчетов по флагу
                dev->cvProcess.wait(lk, [dev] {return dev->process; });
                lk.unlock();
            }

            return PyLong_FromLong(0);
        }
        cerr << "Incorrect id " << id << " in " << devices.size() << " devices" << endl;
        return PyLong_FromLong(-1);
    }

    cerr << "Incorrect args" << endl;
    return PyLong_FromLong(-1);
}

PyObject* init(PyObject*, PyObject* o) {

    //После завершения расчетных функций, нужно задуматься и о quit() и об reinit() - для полноты картины, тут и деструкторы понадобятся.
    //Здесь - проверки, проверки и создание объекта, если все ок
    //Возвращаает в питон id

    if (PyTuple_GET_SIZE(o) == 5) {

        PyArrayObject* const pyPolyline_ = (PyArrayObject*)PyTuple_GetItem(o, 0);
        PyArrayObject* const pyCheckList_ = (PyArrayObject*)PyTuple_GetItem(o, 1);

        PyArrayObject* const pyMaxNlines_ = (PyArrayObject*)PyTuple_GetItem(o, 2);
        PyArrayObject* const pyNlines_ = (PyArrayObject*)PyTuple_GetItem(o, 3);

        PyObject* pyParams = PyTuple_GetItem(o, 4);

        if (PyArray_NDIM(pyPolyline_) != 2 &&
            PyArray_NDIM(pyCheckList_) != 2)
        {
            cerr << "Wrong data dimensions" << endl;
            return PyLong_FromLong(-1);
        }

        if (!PyTuple_Check(pyParams)) {
            cerr << "Wrong parameters arg - must be a tuple" << endl;
            return PyLong_FromLong(-1);
        }

        if (PyErr_Occurred()) {
            cerr << "Bad arguments" << endl;
            PyErr_Clear();
            return PyLong_FromLong(-1);
        }

        try {
            devices.push_back(unique_ptr<Device>(new Device(pyPolyline_, pyCheckList_, pyMaxNlines_, pyNlines_, pyParams, devices.size())));
            return PyLong_FromLongLong(devices.size() - 1);
        }
        catch (exception const*) {
            return PyLong_FromLong(-1);
        }
    }

    cerr << "Incorrect args number" << endl;
    return PyLong_FromLong(-1);
}

static PyMethodDef polylineCpp_methods[] = {
    { "init", (PyCFunction)init, METH_VARARGS, ""},
    { "calcLines", (PyCFunction)pyFun<&Device::closest_pnt>, METH_VARARGS, "Non-blocking!"},
    { "calcLines", (PyCFunction)pyFun<&Device::closest_pnt00>, METH_VARARGS, "Non-blocking!"},
    { "calcLines", (PyCFunction)pyFun<&Device::check_segment_intersections>, METH_VARARGS, "Non-blocking!"},
    { "calcLines", (PyCFunction)pyFun<&Device::check_segment_intersections00>, METH_VARARGS, "Non-blocking!"},
    { "calcLines", (PyCFunction)pyFun<&Device::check_if_obb_intersection>, METH_VARARGS, "Non-blocking!"},
    { "synchronize", (PyCFunction)pyFun<nullptr>, METH_VARARGS, "Waits for all the previous calls to return or returns immediately"},
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef polylineCpp_module = {
    PyModuleDef_HEAD_INIT,
    "polylineCpp",                        // Module name to use with Python import statements
    "Polyline activity",                    // Module description
    0,
    polylineCpp_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_polylineCpp() {
    return PyModule_Create(&polylineCpp_module);
}