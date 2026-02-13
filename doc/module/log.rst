.. _module/log:

Log
###

*vortex* uses the *spdlog* library for high-performance logging in C++.
This module provides a simplified Python wrapper for *spdlog* so that Python users of *vortex* can configure logging.

.. admonition:: C++

    This module exists only as Python bindings.
    C++ users may configure *spdlog* directly according to its `documentation <https://github.com/gabime/spdlog/wiki/QuickStart>`__.

.. _spdlog: https://github.com/gabime/spdlog

Overview
========

The log module exposes logger objects and sinks.
Much like the Python logging framework, loggers create records with a configured name whereas sinks format and process those records.
Logger objects have a severity level to determine which events produce records.

.. note::

    *vortex* configures *spdlog* for asynchronous record sinking in a single background thread when compiled in release mode.
    This scheme offloads handling from critical threads and preserves ordering of log records, but temporal ordering with respect to other console output is not guaranteed.

-   :class:`~vortex.log.Logger`
-   :class:`~vortex.log.StdOutSink`
-   :class:`~vortex.log.StdErrSink`
-   :class:`~vortex.log.ConsoleSink`
-   :class:`~vortex.log.FileSink`
-   :class:`~vortex.log.PythonSink`

Loggers
=======

Create a :class:`~vortex.log.Logger` directly or use a :ref:`helper function <log-helpers>` below.

.. class:: vortex.log.Logger

    Loggers create records and then route them to associated sinks.

    .. method:: __init__(name, sink=None, sinks=None)

        Create a logger with the given name and optionally attack the provided sink/sinks.
        Sinks may be added or removed after the logger is created.

        :param str name:
            The name of the logger
        :param vortex.log.Sink sink:
            A single sink to attach to this logger.
            Mutually exclusive with ``sinks`` argument.
        :param [vortex.log.Sink] sinks:
            A list of sinks to attach to this logger.
            Mutually exclusive with ``sink`` argument.

    .. property:: name
        :type: str

        Name of this logger.
        Read only.

    .. property:: level
        :type: int

        Severity level for events to produce a record.

    .. property:: sinks
        :type: [vortex.log.Sink]

        List of attached sinks.
        Read-only.

        .. caution::

            This property is not thread-safe.
            Do not read this property when adding or removing sinks elsewhere.

    .. method:: set_pattern(pattern)

        Change the pattern for this logger.
        The pattern is applied to all attached sinks as well.
        See the *spdlog* `reference <https://github.com/gabime/spdlog/wiki/Custom-formatting#pattern-flags>`__ for pattern fields.
        The default pattern is ``d-%b-%Y %H:%M:%S.%f] %-10n %^(%L) %v%$``.

        :param str pattern:

            The new pattern.

    .. method:: add_sink(sink)

        Attach a sink to this logger.
        Attaching a sink that is already attached has no effect.

        .. caution::

            This method is not thread-safe.
            Do not call this method when a log record might be emitted (e.g., when the engine is running) or while manipulating the list of attached sinks elsewhere.

        :param vortex.log.Sink sink:

            The sink to attach.

    .. method:: remove_sink(sink)

        Remove a sink from this logger.
        Removing a sink that is not attached has no effect.

        .. caution::

            This method is not thread-safe.
            Do not call this method when a log record might be emitted (e.g., when the engine is running) or while manipulating the list of attached sinks elsewhere.

    .. method:: log(level, msg)

        Emit a log record with the given severity level.
        See *spdlog*'s `source <https://github.com/gabime/spdlog/blob/v1.x/include/spdlog/common.h#L231>`__ for further details.

        ======== =====
        Level    Value
        ======== =====
        trace    0
        debug    1
        info     2
        warn     3
        error    4
        critical 5
        off      6
        ======== =====

        :param int level:

            Severity level of the message.


        :param str msg:

            Message to log.

    .. method:: trace(msg)

        Log a message at the ``trace`` level.

        :param str msg:

            Message to log.

    .. method:: debug(msg)

        Log a message at the ``debug`` level.

        :param str msg:

            Message to log.

    .. method:: info(msg)

        Log a message at the ``info`` level.

        :param str msg:

            Message to log.

    .. method:: warn(msg)

        Log a message at the ``warn`` level.

        :param str msg:

            Message to log.

    .. method:: error(msg)

        Log a message at the ``error`` level.

        :param str msg:

            Message to log.

    .. method:: critical(msg)

        Log a message at the ``critical`` level.

        :param str msg:

            Message to log.

    .. method:: flush()

        Flush all attached sinks.

    .. method:: clone(name)

        Copy the current logger.
        This logger and its copy have the same sinks attached.

        :param str name:

            Name of the copied logger.

        :return vortex.log.Logger:

            The copy.

Sinks
=====

Sinks receive log records and route them for further processing.

.. class:: vortex.log.Sink

    Abstract base class for all sinks.

    .. property:: level
        :type: int

        Severity level for log records undergo processing.

    .. method:: set_pattern(pattern)

        Set the formatting pattern for this sink.
        See :meth:`Logger.set_pattern <vortex.log.Logger.set_pattern>`.

        .. note::

            Calling :meth:`Logger.set_pattern <vortex.log.Logger.set_pattern>` propagates the new pattern to all attached sinks, overwriting any local changes by this method.

    .. method:: flush()

        Ensure all log records have completed processing.

.. class:: vortex.log.StdOutSink

    Base: :class:`~vortex.log.Sink`

    A sink that formats log records to standard out.

    All members are inherited.

.. class:: vortex.log.StdErrSink

    Base: :class:`~vortex.log.Sink`

    A sink that formats log records to standard error.

    All members are inherited.

.. class:: vortex.log.ConsoleSink

    An alias for :class:`~vortex.log.StdErrSink`.

.. class:: vortex.log.FileSink

    Base: :class:`~vortex.log.Sink`

    A sink that formats log records to a file.

    .. property:: filename
        :type: str

        Path of file to store log output.
        Read-only.

    .. method:: change(filename, truncate=False)

        Change the file path of the sink.

        .. note::

            This method is thread-safe.

        :param str filename:

            The new file path.

        :param bool truncate:

            Erase the new file upon opening it if ``True``.

.. class:: vortex.log.PythonSink

    Base: :class:`~vortex.log.Sink`

    A sink that injects log records into the Python logging system.
    Internally, this sink creates and caches a Python :class:`logging.Logger` of the appropriate name and forwards :class:`logging.LogRecord`\ s to :meth:`logging.Logger.handle`.
    Severity levels are translated from *spdlog* to Python after by multiplying their numeric value by multiplying by ``10``.
    The log scope is the name of the :class:`~vortex.log.Logger`.

    +------------------+------------------+
    | *spdlog*         | Python           |
    +----------+-------+----------+-------+
    | Level    | Value | Level    | Value |
    +==========+=======+==========+=======+
    | trace    | 0     | NOTSET   | 0     |
    +----------+-------+----------+-------+
    | debug    | 1     | DEBUG    | 10    |
    +----------+-------+----------+-------+
    | info     | 2     | INFO     | 20    |
    +----------+-------+----------+-------+
    | warn     | 3     | WARNING  | 30    |
    +----------+-------+----------+-------+
    | error    | 4     | ERROR    | 40    |
    +----------+-------+----------+-------+
    | critical | 5     | CRITICAL | 50    |
    +----------+-------+----------+-------+
    | off      | 6     |          |       |
    +----------+-------+----------+-------+

    .. attention::

        Avoid configuring this sink to accept high-frequency log output (e.g., ``trace``-level) as this may degrade logging or application performance.

    .. note::

        It is most efficient to create a single application-wide :class:`~vortex.log.PythonSink` since this avoid duplication of Python :class:`logging.Logger`\ s.


.. _log-helpers:

Helpers
=======

These functions create a logger and sink with a single function call, which is helpful for simple applications or debugging.

.. note::

    These functions are inefficient for larger applications since a new, independent sink is created each call.
    Larger applications should instead prefer to create sinks manually and attach all required loggers.

.. attention::

    In earlier undocumented versions, these functions returned previously-created loggers if a logger of the same name was requested.
    This is no longer the case since a new logger and sink pair is created during each call.

.. function:: vortex.get_console_logger(name, level=2)

    Create a :class:`~vortex.log.Logger` using the given name and level and attach a new :class:`~vortex.log.StdErrSink` to it.

    :return vortex.log.Logger:

        The new logger.

.. function:: vortex.get_file_logger(name, filename, level=2)

    Create a :class:`~vortex.log.Logger` using the given name and level and attach a new :class:`~vortex.log.FileSink` with the given filename.

    :return vortex.log.Logger:

        The new logger.

.. function:: vortex.get_python_logger(name, scope="vortex", level=2)

    Create a :class:`~vortex.log.Logger` using the given name and level and attach a new :class:`~vortex.log.PythonSink` with the given scope.
    The logger's resulting name is ``{scope}.{name}``.

    :return vortex.log.Logger:

        The new logger.
