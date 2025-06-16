/*
 * Event name singletons for ijson's C backend
 *
 * Contributed by Rodrigo Tobar <rtobar@icrar.org>
 *
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia, 2024
 * Copyright by UWA (in the framework of the ICRAR)
 */

#ifndef EVENT_NAMES_H
#define EVENT_NAMES_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define N_EVENTS 11

/*
 * A structure (and variable) holding utf-8 strings with the event names
 * This way we avoid calculating them every time, and we can compare them
 * via direct equality comparison instead of via strcmp.
 * In other words, this is our own idea of string interning.
 */
typedef struct _event_names {
	PyObject *null_ename;
	PyObject *boolean_ename;
	PyObject *integer_ename;
	PyObject *double_ename;
	PyObject *number_ename;
	PyObject *string_ename;
	PyObject *start_map_ename;
	PyObject *map_key_ename;
	PyObject *end_map_ename;
	PyObject *start_array_ename;
	PyObject *end_array_ename;
	Py_hash_t hashes[N_EVENTS]; /* same order as before */
} enames_t;

#define FOR_EACH_EVENT(f)                   \
   f(0, null_ename, "null");                \
   f(1, boolean_ename, "boolean");          \
   f(2, integer_ename, "integer");          \
   f(3, double_ename, "double");            \
   f(4, number_ename, "number");            \
   f(5, string_ename, "string");            \
   f(6, start_map_ename, "start_map");      \
   f(7, map_key_ename, "map_key");          \
   f(8, end_map_ename, "end_map");          \
   f(9, start_array_ename, "start_array");  \
   f(10, end_array_ename, "end_array");

/* Returns the module-internal event name unicode object for the given */
static inline PyObject *get_builtin_ename(enames_t *enames, PyObject *event)
{
#define SWAP(x, y) { Py_INCREF(y); Py_DECREF(x); return y; }

	/* Compare by pointer equality first, then by hash */
#define MATCH(i, member, _value) if (enames->member == event) SWAP(event, enames->member)
	FOR_EACH_EVENT(MATCH)
#undef MATCH

	Py_hash_t hash = PyObject_Hash(event);
	Py_hash_t *hashes = enames->hashes;
#define MATCH(i, member, _value) if (hashes[i] == hash) SWAP(event, enames->member)
	FOR_EACH_EVENT(MATCH)
#undef MATCH

	return event;
}

#endif /* EVENT_NAMES_H */