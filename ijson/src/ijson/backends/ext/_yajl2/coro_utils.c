/*
 * Coroutine utilities implementation for ijson's C backend
 *
 * Contributed by Rodrigo Tobar <rtobar@icrar.org>
 *
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia, 2020
 * Copyright by UWA (in the framework of the ICRAR)
 */

#include "common.h"
#include "coro_utils.h"

static PyObject * create_coro(PyObject *target, pipeline_node node)
{
	PyObject *coro_args;
	if (node.args) {
		int nargs = PyTuple_Size(node.args);
		N_N(coro_args = PyTuple_New(nargs + 1));
		Py_INCREF(target);
		PyTuple_SET_ITEM(coro_args, 0, target);
		int i;
		for (i = 0; i != nargs; i++) {
			PyTuple_SET_ITEM(coro_args, i + 1, PySequence_GetItem(node.args, i));
		}
	}
	else {
		N_N(coro_args = PyTuple_Pack(1, target));
	}
	PyObject *coro = PyObject_Call((PyObject *)node.type, coro_args, node.kwargs);
	Py_DECREF(coro_args);
	return coro;
}

PyObject *chain(PyObject *sink, pipeline_node *coro_pipeline)
{
	PyObject *coro = NULL;
	PyObject *prev = sink;
	int element = 0;
	Py_INCREF(prev);
	while (1) {
		pipeline_node node = coro_pipeline[element++];
		if (node.type == NULL) {
			break;
		}
		coro = create_coro(prev, node);
		Py_DECREF(prev);
		N_N(coro);
		prev = coro;
	}
	return coro;
}