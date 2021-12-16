
import * as d3 from "d3";

function HistogramChart() {
  let xAttr = 'x';
  let width = 200, height = 30
  let margin = { top: 0, bottom: 0, left: 0, right: 0 };
  let color = d3.schemeDark2[0];
  let datatype = "category"
  let data = null
  const maxbins = 40
  let valueTicks = null, valueScale
  let on_mousemove = () => {}
  let on_mouseout = () => {}

  HistogramChart.prototype.cache = {}

  function chart(context) {
    let selection = context.selection ? context.selection() : context
    const g = selection
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`)

    if (!valueTicks) {
      if (datatype == "time") {
        const extent = d3.extent(data, (d) => new Date(d[xAttr]));
        valueScale = d3.scaleTime().domain(extent).nice();
        valueTicks = valueScale.ticks(maxbins);
        valueScale.range([0, valueTicks.length]);
      } else if (datatype == "number") {
        const extent = d3.extent(data, (d) => d[xAttr]);
        valueScale = d3.scaleLinear().domain(extent).nice();
        valueTicks = valueScale.ticks(maxbins);
        valueScale.range([0, valueTicks.length]);
      } else {
        valueTicks = [...new Set(data.map((d) => d[xAttr]))];
        const dict = {};
        valueTicks.forEach((d, i) => (dict[d] = i));
        valueScale = (d) => dict[d];
      }
    } else {
      if (datatype == "time") {
        const extent = d3.extent(data, (d) => new Date(d[xAttr]));
        valueScale = d3.scaleTime().domain(extent).nice();
        valueScale.range([0, valueTicks.length]);
      } else if (datatype == "number") {
        const extent = valueTicks
        valueScale = d3.scaleLinear().domain(extent).nice();
        valueTicks = valueScale.ticks(maxbins);
        valueScale.range([0, valueTicks.length]);
      } else {
        const dict = {};
        valueTicks.forEach((d, i) => (dict[d] = i));
        valueScale = (d) => dict[d]
      }
    }

    const summary_data = valueTicks.map(name => ({ name, count: 0 }))
    data.forEach(d => {
      summary_data[Math.min(valueTicks.length - 1, ~~valueScale(d[xAttr]))].count += 1
    })

    const xScale = d3
      .scaleBand()
      .domain(valueTicks)
      .range([0, width])
      .padding(0.2);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(summary_data, d => d.count)])
      .range([0, height])

    // x axis
    g.append("g").attr("transform", `translate(0, ${height + 5})`);
    //.call(xAxis)

    // Bars
    const bar = g.selectAll("rect")
      .data(summary_data)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d) => xScale(d.name))
      .attr("y", (d) => height - yScale(d.count))
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => yScale(d.count))
      .attr("fill", color)
      .attr("opacity", 1)
  }

  function functor(x) {
    return typeof x === "function"
      ? x
      : function () {
          return x;
        };
  }

  chart.x = function (_) {
    if (!arguments.length) return xAttr;
    xAttr = _;
    return chart;
  };

  chart.width = function (_) {
    if (!arguments.length) return width;
    width = _;
    return chart;
  };

  chart.height = function (_) {
    if (!arguments.length) return height;
    height = _;
    return chart;
  };

  chart.margin = function (_) {
    if (!arguments.length) return margin;
    margin = _;
    return chart;
  };

  chart.datatype = function (_) {
    if (!arguments.length) return datatype;
    datatype = _;
    return chart;
  };

  chart.color = function (_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };

  chart.data = function (_) {
    if (!arguments.length) return data;
    data = _;
    return chart;
  };

  chart.brushend = function (_) {
    if (!arguments.length) return on_brushend;
    on_brushend = _;
    return chart;
  };

  chart.mousemove = function (_) {
    if (!arguments.length) return on_mousemove;
    on_mousemove = _;
    return chart;
  };

  chart.mouseout = function (_) {
    if (!arguments.length) return on_mouseout;
    on_mouseout = _;
    return chart;
  };

  chart.valueTicks = function(_) {
    if (!arguments.length) return valueTicks
    valueTicks = _
    return chart
  }

  return chart;
}

export default HistogramChart
