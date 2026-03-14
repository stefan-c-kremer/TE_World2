# TEWorldCodeV2 Graphing

There are several graphing functionalities of this repository, which are outlined below.

## `CreateTraceGraphs`: Quick Graphing of Individual Trials

[`CreateTraceGraphs.py`](CreateTraceGraphs.py) is a quick and easy-to-use program that takes CSV trace files that were produced by a list of trials, and outputs corresponding graphs that are predefined in the code. These include:

- Host Population vs Generation
- Total Live TEs vs Generation
- Live TE Percentiles vs Generation
- Total Dead TEs vs Generation
- Dead TE Percentiles vs Generation
- Fitness Percentiles vs Generation
- TE Deaths vs Generation
- TE Collisions vs Generation
- TE Jumps vs Generation
- TE Jump Effects vs Generation
- TE and Gene Locations
- Live autonomous/non-autonomous TEs vs Generation

### Run `CreateTraceGraphs`

1. Navigate to the same folder as `CreateTraceGraphs.py` within your terminal.

2. Copy any trace files that you want to graph into this directory. They will be identified by the code. Ensure that they match the regular expression `trace-[0-9]{3}.csv` (i.e. `trace-001.csv`).

3. Run `CreateTraceGraphs.py`.

```
python3 CreateTraceGraphs.py
```

The graphs should be created and output as `.svg` files, for all of the specified trace files.

4. To delete all the HTML and SVG files, use the Makefile.

```
make
```