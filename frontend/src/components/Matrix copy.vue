<template>
  <div class="matrix-container" ref="matrix_parent">
    <svg class="matrixdiagram">
      <g class="header_canvas" :transform="`translate(${setting.margin.left},${setting.margin.top})`"></g>
      <g class="rule_canvas" :transform="`translate(${setting.margin.left},${setting.margin.top})`"></g>
    </svg>
  </div>
</template>

<script>
import { mapActions, mapState } from 'vuex'
import * as d3 from 'd3'
import glyph from '../libs/glyph'

export default {
  name: 'Matrix',
  data() {
    return {
      current_col: null,
      current_row: null,
    }
  },
  computed: {
    ...mapState([ 'setting', 'layout', 'primary', 'is_zoomed' ])
  },
  watch: {
    layout(val) {
      if (val != null) {
        this.render()
      }
    }
  },
  beforeDestroy () {
    if (typeof window === 'undefined') return
    window.removeEventListener('resize', this.onResize, { passive: true })
  },
  async mounted() {
    window.addEventListener('resize', this.onResize, { passive: true })
    this.onResize()
  },
  methods: {
    ...mapActions([ 'orderColumn', 'showRepresentRules', 'showExploreRules', 'updateWidth' ]),
    onResize(){
      const width = this.$refs.matrix_parent.getBoundingClientRect().width
      this.updateWidth(width)
    },
    render() {
      const self = this
      // const min_confidence = 5
      const setting = this.setting
      const { margin, width, height } = setting
      const header_offset = { x: 105, y: 45 } //this.primary.has_primary_key ? 20 : 5 }

      const svg = d3.select(".matrixdiagram")
        .attr('width', width)
        .attr('height', height)

      /*<svg style="width:24px;height:24px" viewBox="0 0 24 24">
    <path fill="currentColor" d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z" />
</svg>*/
      svg
        .append("symbol")
        .attr("id", "markermore")
        .attr("viewBox", "0 0 24 24")
        .append("path")
        //.style("fill", "#333")
        .attr("d", "M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z")

      svg
        .append("symbol")
        .attr("id", "markerexpand")
        .attr("viewBox", "0 0 24 24")
        .append("path")
        //.style("fill", "#333")
        .attr("d", "M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z")

      svg
        .append("symbol")
        .attr("id", "markercollapse")
        .attr("viewBox", "0 0 24 24")
        .append("path")
        //.style("fill", "#333")
        .attr("d", "M7.41,15.41L12,10.83L16.59,15.41L18,14L12,8L6,14L7.41,15.41Z")
      
      const tooltip = d3.select(".svg-tooltip")
        .text("I'm a circle!")
    
      const header_canvas = svg.select(".header_canvas")
      const rule_canvas = svg.select(".rule_canvas")
      
      const layout = this.layout
      
      function brushed({selection}) {
        if (self.is_zoomed) {
          self.showRepresentRules()
        } else {
          const selected_rules = layout.rows
            .filter(d => d.y >= selection[0] && d.y + d.height <= selection[1])
            .map(d => d.rule.id)
          self.showExploreRules(selected_rules)
        }
          /*
        if (is_zoomed) {
          if (!selection) return
          is_zoomed = 0
          selected_data = selectRules(middle_data)
        } else {
          if (!selection) return
          is_zoomed = 1
          const col = d3.select(this).datum()
          selected_data.rules = middle_data.raw_rules
            .filter(d => {
              if (!d.cond_dict[col.index]) {
                return false
              }
              let x1 = col.scale(d.cond_dict[col.index][0]) - col.x
              let x2 = col.scale(d.cond_dict[col.index][1]) - col.x
              return x1 >= selection[0][0] && x2 <= selection[1][0] && d.fidelity > min_confidence
            })
        }
        const newLayout = calcLayout({ rules: selected_data, instances: rawdata.X })
        layout.rows = newLayout.rows
        update()
        // d3.select(this).call(brush.move, null)
          */
      }
      
      const xrange = [Math.min(...layout.cols.map(d => d.x)), Math.max(...layout.cols.map(d => d.x)) + setting.coverage_width]
      const yrange = [Math.min(...layout.rows.map(d => d.y)), Math.max(...layout.rows.map(d => d.y + d.height))]

      function updateCols() {
        const cols_data = layout.cols
        // console.log('layout.cols', layout.cols)
        function dragstarted(event, d) {
          d3.select(this).raise()
          d3.select(this).select(".header").attr("stroke", "black")
          d3.select(this).select(".background").attr("stroke", "black")
        }

        function dragged(event, d) {
          // d.x = event.x
          // d.y = event.y
          d3.select(this).attr("transform", `translate(${event.x},${d.y})`)
        }

        function dragended(event, d) {
          d3.select(this).select(".header").attr("stroke", null)
          d3.select(this).select(".background").attr("stroke", null)
        }

        const drag = d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        
        let col = header_canvas.selectAll('g.col')
          .data(cols_data, d => d.index)

        let col_join = col.enter().append('g')
          .attr('class', 'col')
        col.exit().selectAll('*').remove()

        col = header_canvas.selectAll('g.col')
          .data(cols_data, d => d.index)
          .call(drag)

        col_join.append('path').attr('class', 'header')
        col_join.append('text').attr('class', 'label')
        col_join.append('g').attr('class', 'axis')

        const col_background = col_join
          .append('g').attr('class', 'col-content')
        col_background.append('rect').attr('class', 'background')
        //col_background
        //  .filter(d => !new_cols_x[d.index])
        //  .append('g').attr('class', 'brush')
        
        function headerInteraction(x) {
          x.on('mouseover', function(){
            d3.select(this.parentNode).select('rect.background').attr('stroke', 'black').attr('stroke-width', 1)
            d3.select(this.parentNode).select('path.header').attr('fill-opacity', .8).attr('stroke-width', 1)
              .attr('stroke', 'black')
          }).on('mouseout', function(){
            d3.select(this.parentNode).select('rect.background').attr('stroke', 'none')
            d3.select(this.parentNode).select('path.header').attr('fill-opacity', .3).attr('stroke-width', .3)
              .attr('stroke', 'lightgray')
          }).on('click', function(ev, d) {
            self.orderColumn(d.index)
          })
        }

        col.select('path.header')
          .attr('d', d => {
            return `M0,0 L${d.width},0 L${d.width},${-header_offset.y} L${d.width+header_offset.x*1.5},${-header_offset.y-header_offset.x}
                         L${header_offset.x*1.5},${-header_offset.y-header_offset.x} L${0},${-header_offset.y} z`
          })
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)
          .call(headerInteraction)
        
        col.select('text.label')
          .attr('transform', `rotate(-35)`)
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .attr('dx', d => d.index == self.primary.key ? 18: 27)
          .attr('dy', 24 - header_offset.y)
          .text(d => {
            let prefix = ''
            if (d.index == self.primary.key) {
              prefix = self.primary.order == 1 ? '▲' : '▼'
            }
            let name = d.name
            if (d.name.length > 30) {
              for (let i = name.length - 1; i >= 0; --i) {
                if (name[i] == ' ') {
                  name = name.slice(0, i)
                  break
                }
              }
            }
            return prefix + name
          })
          .call(headerInteraction)

        col.select('rect.background')
          .attr('width', d => d.width)
          .attr('height', d => d.height)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', setting.cell_stroke_width)
          .attr('fill', 'white')
        
        col.filter(d => !d.show_axis).each(function(d){
          d3.select(this).select('g.axis').selectAll('*').remove()
        })
        col.filter(d => d.show_axis).each(function(d){
          d3.select(this).select('g.axis')
            //.attr('transform', `translate(0,${-header_offset.y + 5})`)
            .call(d3.axisTop(d.scale).ticks(4))
          d3.select(this).select('g.axis').raise()
        })
        
        col
          //.transition().duration(setting.duration)
          .attr('transform', d => `translate(${d.x},${d.y})`)
      }
      
      function updateRows() {
        rule_canvas.selectAll("g.brush").remove()
        let canvas_brush = rule_canvas
          .append('g').attr('class', 'brush')

        // console.log(layout.rows.map(d => d.rule.represent))
        let row = rule_canvas.selectAll('g.row')
          .data(layout.rows, d => d.rule.id)
        
        row.exit()
          .style('opacity', 0)
          .selectAll('*').remove()

        let instance = rule_canvas.selectAll('g.instance')
          .data(layout.instances, d => d.id)
        
        instance.exit()
          .style('opacity', 0)
          .selectAll('*').remove()
        
        let row_join = row.enter().append('g')
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .attr('class', 'row')
          .style('opacity', 0)

        row.exit().selectAll('*').remove()

        /*
        row_join.append('path')
          .attr('class', 'glyph')
          .attr('d', glyph.star.path)
          .attr('transform', `translate(0, ${-1}) scale(${8 / glyph.star.size})`)
          .attr('fill', 'gray')
          .attr('opacity', 0)
        */
        row_join.append('g')
          .attr('class', 'glyph')
          .attr('transform', `translate(-60, 0)`)
        /*
          .attr('d', glyph.star.path)
          .attr('transform', `translate(0, ${-1}) scale(${8 / glyph.star.size})`)
          .attr('fill', 'gray')
          .attr('opacity', 0)

        row_join.append('rect')
          .attr('class', 'background')
          .attr('x', 35)
          .attr('width', d => d.width + 80)
          .attr('height', d => d.height)
          .attr('fill', 'yellow')
        */

        //row_join.select('.glyph').selectAll('*').remove()

        let represent_glyph = row_join.filter(d => d.rule.represent)
          .select('.glyph')
          //.attr("opacity", .5)
        
        /*
        represent_glyph
          .append('line')
          .attr('x1', 10)
          .attr('x2', 80)
          .attr('y1', d => d.height / 2)
          .attr('y2', d => d.height / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke_width', '2px')
        */

        represent_glyph
          .append('circle')
          .attr('cx', 90)
          .attr('cy', d => d.height / 2)
          .attr('r', d => d.height / 2 - 2)
          .attr('fill', 'darkgray')
          .attr('display', 'none')

        let rep_glyph = represent_glyph
          .append('g')
          .attr('class', 'represent_glyph')
          //.attr('opacity', 0.5)
          //.on('mouseover', function(){ d3.select(this).transition().duration(300).attr('opacity', 1) })
          //.on('mouseout', function(){ d3.select(this).transition().duration(300).attr('opacity', .5) })
/*
        rep_glyph
          .append('rect')
          .attr('x', 15)
          .attr('y', 1)
          .attr('width', d => d.height * 4)//d.attr.num)
          .attr('height', d => d.height - 2)
          .attr('fill', d => d3.interpolateLab('white', 'darkgray')(
            ~~(d.attr.num * 5) / 5
          ))
          .attr('stroke', 'none')
          .attr('stroke_width', '1.5px')
*/
        represent_glyph
          .each(function(d){
            const n = ~~d.attr.num
            for (let i = 0; i < n; ++i) {
              d3.select(this)
                .append("circle")
                .attr("cx", d => 90 - (n - i) * 4 - d.height / 2)
                .attr("cy", d => d.height / 2)
                .attr("r", d => d.height / 2 - 1)
                .attr("fill", i == n - 1 ? "darkgray" : "white")
                .attr("stroke", "darkgray")
                .attr("stroke-width", 1)
            }
          })

        rep_glyph
          .append("use")
          .attr("href", self.is_zoomed ? "#markercollapse" : "#markerexpand")
          .attr("x", 20 -2)
          .attr("width", 12)
          .attr("height", 12)
          .attr("y", -1.5)
          .style("fill", "darkgray")

        let represent_glyph_dot = rep_glyph
          .append("g")
          .attr("class", "dot")
          .attr("display", "none")

        represent_glyph_dot
          .append("rect")
          .attr("x", 12)
          .attr("y", d => d.height / 2 - 1)
          .attr("height", 2)
          .attr("width", 2)
          .attr("fill", "darkgray")

        represent_glyph_dot
          .append("rect")
          .attr("x", 16)
          .attr("y", d => d.height / 2 - 1)
          .attr("height", 2)
          .attr("width", 2)
          .attr("fill", "darkgray")

        represent_glyph_dot
          .append("rect")
          .attr("x", 20)
          .attr("y", d => d.height / 2 - 1)
          .attr("height", 2)
          .attr("width", 2)
          .attr("fill", "darkgray")

        let nonrepresent_glyph = row_join.filter(d => !d.rule.represent)
          .select('.glyph')
          //.attr("opacity", .5)

        nonrepresent_glyph
          .append('line')
          .attr('x1', 50)
          .attr('x2', 90)
          .attr('y1', d => d.height / 2)
          .attr('y2', d => d.height / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke_width', '2px')

        nonrepresent_glyph
          .append('circle')
          .attr('cx', 90)
          .attr('cy', d => d.height / 2)
          .attr('r', d => d.height / 2 - 2)
          .attr('fill', 'darkgray')
          .attr('stroke', 'none')

        nonrepresent_glyph
          .append('line')
          .attr('x1', 50)
          .attr('x2', 50)
          .attr('y1', d => -d.height / 2)
          .attr('y2', d => d.height / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke_width', '2px')
        
        row_join = row_join.merge(row)
        
        const brush = d3.brush()
          .on("end", brushed)

        canvas_brush
          .call(
            d3.brushY().extent([[xrange[0] + 10, yrange[0]], [xrange[1] + setting.coverage_width + 10, yrange[1]]]).on("end", brushed)
          )

        let row_extend = row_join.selectAll('g.extended')
          .data(d => d.extends)
          .enter().append('g')
          .attr('class', 'extended')
          .attr('opacity', 1)
        row_extend.exit().selectAll('*').remove()
        
        let cell = row_join.selectAll('g.cell')
          .data(d => d.items)
          .enter().append('g')
          .attr('class', 'cell')
          .attr('opacity', 1)
        cell.exit().selectAll('*').remove()
  
        row_extend.append('rect').attr('class', 'bar')
        cell.append('rect').attr('class', 'bar')
        cell.append('rect').attr('class', 'barbg')

        row = row.merge(row_join)
        cell = row.selectAll('g.cell')
        row_extend = row.selectAll('g.extended')
/*
        row.selectAll('path.glyph')
          .transition().duration(setting.duration)
          .attr('opacity', d => self.is_zoomed ? (d.rule.represent ? 1 : 0) : 0)

        row.selectAll('rect.background')
          .transition().duration(setting.duration)
          .attr('opacity', d => self.is_zoomed ? (d.rule.represent ? 1 : 0) : 0)
*/
        cell.select('rect.barbg')
          .attr('x', d => d.x)
          .attr('width', d => d.width)
          .attr('height', d => d.height - setting.cell_padding)
          .attr('fill', d => d3.interpolateLab('white', d.fill)(0.25))
          .attr('opacity', 1)

        canvas_brush.raise()

        cell.select('rect.bar')
          .on("mouseover", function(){
            return tooltip.style("visibility", "visible")
          })
          .on("mouseout", function(){
            return tooltip.style("visibility", "hidden")
          })
          .on("mousemove", function(ev, d){
            tooltip.text(`[${Number(d.cond.range[0]).toFixed(3)},${Number(d.cond.range[1]).toFixed(3)}]`)
            return tooltip.style("top", (ev.pageY-10)+"px").style("left",(ev.pageX+10)+"px")
          })
          .raise()
          
        cell.select('rect.bar')
          .transition().duration(setting.duration)
          .attr('x', d => d.x1)
          .attr('width', d => d.x2 - d.x1)
          .attr('height', d => d.height - setting.cell_padding)
          .attr('fill', d => d.fill)

        row_extend.select('rect.bar')
          .on("mouseover", function(){
            return tooltip.style("visibility", "visible")
          })
          .on("mouseout", function(){
            return tooltip.style("visibility", "hidden")
          })
          .on("mousemove", function(ev, d){
            tooltip.text(`${Number(d.value).toFixed(3)}`)
            return tooltip.style("top", (ev.pageY-10)+"px").style("left",(ev.pageX+10)+"px")
          })
          .raise()

        row_extend.select('rect.bar')
          .attr('x', d => d.x1)
          .attr('width', d => d.x2 - d.x1)
          .attr('height', d => d.height - setting.cell_padding)
          .attr('fill', d => d.fill)
        
        row
          .transition().duration(setting.duration)
          .transition().duration(setting.duration)
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .transition().duration(setting.duration)
          .style('opacity', 1)//d => d.rule.represent ? 1 : 0.5)

/*
        let instance_join = instance.enter().append('g')
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .attr('class', 'instance')
          .style('opacity', 0)
        
        instance_join = instance_join.merge(instance)
        let dim = instance_join.selectAll('g.dim')
          .data(d => d.dims)
          .join('g')
          .attr('class', 'dim')
          .attr('opacity', .9)

        dim.append('circle').attr('class', 'value')
        instance = instance.merge(instance_join)
        dim = instance.selectAll('g.dim')

        dim.select('circle.value')
          .attr('cx', d => d.x)
          .attr('cy', d => d.y)
          .attr('r', d => d.r)
          .attr('fill', d => d.fill)
          .attr('stroke', 'none')
          .style('opacity', .8)

        instance
          .transition().duration(setting.duration)
          .delay(setting.duration)
          .style('opacity', 1)
*/
      }
      
      function update() {
        updateCols()
        updateRows()
      }
      
      update()
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
</style>
