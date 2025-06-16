/*
 * Common definitions for ijson's C backend
 *
 * Contributed by Rodrigo Tobar <rtobar@icrar.org>
 *
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia, 2019
 * Copyright by UWA (in the framework of the ICRAR)
 */

#ifndef COMMON_H
#define COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * Error-handling macros to help reducing clutter in the code.
 * N: NULL, M1: -1, Z: zero, NZ: not-zero, LZ: less-than-zero
 * */
#define RETURN_X_IF_COND(statement, X, cond) \
	do { \
		if ((statement) cond) { \
			return X; \
		} \
	} while(0);
#define M1_M1(stmt)   RETURN_X_IF_COND(stmt,   -1, == -1)
#define M1_N(stmt)    RETURN_X_IF_COND(stmt,   -1, == NULL)
#define M1_NZ(stmt)   RETURN_X_IF_COND(stmt,   -1, != 0)
#define M1_Z(stmt)    RETURN_X_IF_COND(stmt,   -1, == 0)
#define N_M1(stmt)    RETURN_X_IF_COND(stmt, NULL, == -1)
#define N_N(stmt)     RETURN_X_IF_COND(stmt, NULL, == NULL)
#define N_Z(stmt)     RETURN_X_IF_COND(stmt, NULL, == 0)
#define N_NZ(stmt)    RETURN_X_IF_COND(stmt, NULL, != 0)
#define Z_M1(stmt)    RETURN_X_IF_COND(stmt,    0, == -1)
#define Z_N(stmt)     RETURN_X_IF_COND(stmt,    0, == NULL)
#define Z_NZ(stmt)    RETURN_X_IF_COND(stmt,    0, != 0)
#define X_LZ(stmt, X) RETURN_X_IF_COND(stmt,    X, < 0)
#define X_N(stmt, X)  RETURN_X_IF_COND(stmt,    X, == NULL)

#define CORO_SEND(target_send, event) \
	{ \
		if (PyList_Check(target_send)) { \
			Z_M1(PyList_Append(target_send, event)); \
		} \
		else { \
			Z_N( PyObject_CallFunctionObjArgs(target_send, event, NULL) ); \
		} \
	}

/* Common function used by __iter__ method in coroutines/generators */
PyObject* ijson_return_self(PyObject *self);

/* Common function used by empty methods in coroutines/generators */
PyObject* ijson_return_none(PyObject *self);

/* Unpacks an iterable into multiple values, returns 0 on success, -1 on error */
int ijson_unpack(PyObject *o, Py_ssize_t expected, ...);

#endif /* COMMON_H */