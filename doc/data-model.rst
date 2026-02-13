Data Model
==========

*vortex* implements related data models for organizing scan patterns and acquired data in memory.

Scan Patterns
-------------

The fundamental unit of the *vortex* scan model is the **sample**.
Each sample corresponds to a point in the imaging scene at which data is acquired.
Samples are organized hierarchically into segments, volumes, and scans.

A **segment** is a temporally-ordered sequence of samples.
Most frequently a segment represents a line across the imaging scene, although there is no formal geometric constraint.
A **volume** is a logically-ordered collection of segments.
Each segment has a unique index within the volume, but segments may be acquired in any order.
A **scan** is a logically-ordered collection of volumes.
Like segments, each volume has a unique index within the scan.
Volumes may be acquired in any order within a scan.

=======  ========  ===========  ============
Name     Ordering  Subdivision  OCT Analogue
=======  ========  ===========  ============
sample                          A-scan
segment  temporal  sample       B-scan
volume   logical   segment
scan     logical   volume
=======  ========  ===========  ============

Memory Layout
-------------

The fundamental unit of the *vortex* memory model is the **sample**.
This is the smallest unit of data acquired by the an acquisition component.
Each sample may consist of multiple **channels**.
For multi-channel acquisitions, a sample will contain as many channels as the acquisition.
Single-channel acquisition are considered to have one channel per sample.
A **record** is a sequence of samples in order of acquisition, and a **block** is a sequence of records in order of acquisition.

=======  =========  =========  ===========  ============
Name     Dimension  Shape      Subdivision  OCT Analogue
=======  =========  =========  ===========  ============
channel  0D
sample   1D         C          channel
record   2D         S x C      sample       A-scan
block    3D         R x S x C  record       B-scan
=======  =========  =========  ===========  ============

C = channels per sample, S = samples per record, R = records per block

The scan pattern and memory models interface only at the sample and record level, respectively.
Specifically, the data acquired at a given scan pattern sample yields a record in memory.
*vortex* components operate on blocks rather than segments or volumes which decouples the scan pattern from the underlying memory layout.
This provides considerable flexibility in scan pattern generation.

.. important::
    *vortex* uses C-style or row-major data ordering.

Memory Management
-----------------

*vortex* manages its own CPU and GPU memory allocations to avoid memory leaks.
When providing access to buffers via accessors or callbacks, *vortex* maintains ownership of the underlying buffer.
Applications that wish to maintain buffer control should make a copy.
