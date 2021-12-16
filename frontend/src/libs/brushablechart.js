
import * as d3 from "d3";

function BrushableBarchart() {
  let xAttr = 'x';
  let width = 200,
    height = 30;
  let margin = { top: 25, bottom: 25, left: 20, right: 20 };
  let colors = { handle: d3.schemeDark2[1], bar: d3.schemeDark2[0] };
  let datatype = "category"
  let data = null
  const maxbins = 40
  const categorical_thres = 5
  const min_display_height = 3
  const unselect_opacity = .2
  let valueTicks, valueScale
  let filter = (d) => 1
  let on_brushend = () => {}
  let on_mousemove = () => {}
  let on_mouseout = () => {}

  function chart(context) {
    let selection = context.selection ? context.selection() : context
    const g = selection
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`)
    const highlight_bar_color = d3.color(colors.bar).darker(1.5)

    const unique_values = [...new Set(data.map(d => d[xAttr]))]
    if (unique_values.length <= categorical_thres) {
      datatype = 'category'
    } else if (unique_values.length > 15) {
      datatype = 'number'
    }
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
      .domain([0, 1, d3.max(summary_data, d => d.count)])
      .range([0, min_display_height, height])
/*
    const xAxis = d3
      .axisBottom()
      .scale(xScale)
      .tickValues(xScale.domain().filter((d, i) => !(i % 3)));
*/
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
      .attr("fill", colors.bar)
      .attr("opacity", 1)

    g.append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", height)
      .attr("y2", height)
      .attr("stroke", colors.handle)
      .attr("stroke-width", ".5px")

    if (datatype == 'time' || datatype == 'number') {
      const triangle = d3.symbol().size(48).type(d3.symbolTriangle);
  
      const filteredDomain = function (scale, min, max) {
        let dif = scale(d3.min(scale.domain())) - scale.range()[0],
          iMin = min - dif < 0 ? 0 : Math.round((min - dif) / xScale.step()),
          iMax = Math.round((max - dif) / xScale.step());
        if (iMax == iMin) --iMin;
        return scale.domain().slice(iMin, iMax);
      };
  
      const snappedSelection = function (bandScale, domain) {
        const min = d3.min(domain),
          max = d3.max(domain);
        return [bandScale(min), bandScale(max) + bandScale.bandwidth()];
      };

      let last_selection = null
      const brush = d3
        .brushX()
        .handleSize(8)
        .extent([
          [0, 0],
          [width, height],
        ])
        .on("end", function(ev){
          selection
            .select("rect.selection")
            .attr("y", height)
            .attr("height", 4)
            .attr("fill", colors.glyph)
            .attr("fill-opacity", .6)
          on_brushend()
        })
        .on("start brush", function (ev) {
          if (!ev.selection && !ev.sourceEvent) return;
          if (last_selection && last_selection[0] == ev.selection[0] && last_selection[1] == ev.selection[1]) {
            return
          }
          last_selection = ev.selection
          const s0 = ev.selection
              ? ev.selection
              : [1, 2].fill(ev.sourceEvent.offsetX),
            d0 = filteredDomain(xScale, ...s0);
          let s1 = s0;
          if (ev.sourceEvent && ev.type === "end") {
            s1 = snappedSelection(xScale, d0);
            d3.select(this).transition().call(ev.target.move, s1);
          }
          filter = (d) => d[xAttr] >= d0[0] && d[xAttr] <= d0[d0.length - 1]
  
          // move handlers
          selection.selectAll("g.handles").attr("transform", (d) => {
            const x = d == "handle--o" ? s1[0] : s1[1];
            return `translate(${x}, 0)`;
          });

          selection
            .select("rect.selection")
            .attr("y", height)
            .attr("height", 4)
            .attr("fill", colors.glyph)
            .attr("fill-opacity", .6)
  
          // update labels
          selection.selectAll("g.handles")
            .selectAll("text")
            .attr("dx", d0.length > 1 ? 0 : 6)
            .text((d, i) => {
              let year;
              if (d0.length > 1) {
                year = d == "handle--o" ? d3.min(d0) : d3.max(d0);
              } else {
                year = d == "handle--o" ? d3.min(d0) : "";
              }
              return year
            });
  
          // update bars
          bar.attr("opacity", (d) =>
            d0.includes(d.name) ? 1 : unselect_opacity
          );
        });

      const gBrush = g.append("g").call(brush).call(brush.move, [0, width])

      const gHandles = gBrush
        .selectAll("g.handles")
        .data(["handle--o", "handle--e"])
        .enter()
        .append("g")
        .attr("class", (d) => `handles ${d}`)
        .attr("fill", colors.handle)
        .attr("transform", (d) => {
          const x = d == "handle--o" ? 0 : width;
          return `translate(${x}, 0)`;
        })

      const gSelection = gBrush
        .select("rect.selection")
        .attr("y", height)
        .attr("height", 4)
        .attr("fill", colors.glyph)
        .attr("fill-opacity", .6)

      // Label
      gHandles
        .selectAll("text")
        .data((d) => [d])
        .enter()
        .append("text")
        .style("font-size", "12px")
        .attr("text-anchor", "middle")
        .attr("dy", -10)
        .text((d) => d == "handle--o" ? d3.min(xScale.domain()) : d3.max(xScale.domain()));

      // Triangle
      gHandles
        .selectAll(".triangle")
        .data((d) => [d])
        .enter()
        .append("path")
        .attr("class", (d) => `triangle ${d}`)
        .attr("d", triangle)
        .attr('fill', colors.glyph)
        .attr("transform", (d) => {
          const x = d == "handle--o" ? -6 : 6,
            offset = d == "handle--o" ? 6 : -6;
          return `translate(${x + offset}, ${height + 5})`;
        });

      // Visible Line
      gHandles
        .selectAll(".line")
        .data((d) => [d])
        .enter()
        .append("line")
        .attr("class", (d) => `line ${d}`)
        .attr("x1", 0)
        .attr("y1", -3.5)
        .attr("x2", 0)
        .attr("y2", height + 1.5)
        .attr("stroke", colors.handle);
    } else {
      summary_data.forEach(d => d.selected = 1)
      bar.on('mousemove', function(ev, d){
        d3.select(this).attr('fill', highlight_bar_color)
        on_mousemove(ev, d)
      }).on('mouseout', function(ev, d){
        d3.select(this).attr('fill', colors.bar)
        on_mouseout(ev, d)
      }).on('click', function(ev, d){
        d.selected = !d.selected
        d3.select(this).attr('opacity', d.selected ? 1 : unselect_opacity)
        filter = (t) => {
          for (let el of summary_data) {
            if (el.name == t[xAttr]) return el.selected
          }
          return 0
        }
        on_brushend()
      })
    }
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

  chart.colors = function (_) {
    if (!arguments.length) return colors;
    colors = _;
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

  chart.filter = function () {
    return filter;
  };

  return chart;
}

export default BrushableBarchart
