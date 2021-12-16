<template>
  <div class="matrix-container" ref="matrix_parent">
    <!--div class="text-center" v-if="matrixview.order_keys.length > 0">
      Order by:
      <v-btn
        v-for="item in matrixview.order_keys"
        :key="item.key"
        class="ma-2 pa-1"
        close
        @click:close="chip1 = false"
        style="text-transform: none!important"
      >
        {{ item.name }} {{ item.order == 1 ? '' : '(Descending)'}}
      </v-btn>
    </div-->
    <svg class="matrixdiagram">
      <clipPath id="rule_clip">
        <rect :width="`${matrixview.width + 5}`" 
          :height="`${matrixview.height - matrixview.margin.bottom - matrixview.margin.top + 5}`">
        </rect>
      </clipPath>
      <g class="header_container" 
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width},${matrixview.margin.top})`">
      </g>
      <g class="rule_canvas_container"
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width},${matrixview.margin.top})`">
        <g class="rule_outer" clip-path="url(#rule_clip)">
          <g class="rule_canvas" :transform="`translate(0, ${current_scroll})`">
          </g>
        </g>
      </g>
      <g class="status_container" 
        :transform="`translate(${matrixview.margin.left},${matrixview.height - matrixview.margin.bottom})`">
      </g>
      <g class="scrollbar_container"></g>
    </svg>
  </div>
</template>

<script>
import { mapActions, mapGetters, mapState } from 'vuex'
import * as d3 from 'd3'
import HistogramChart from "../libs/histogramchart";
import Scrollbar from "../libs/scrollbar";

export default {
  name: 'Matrix',
  data() {
    return {
      current_col: null,
      current_row: null,
      current_scroll: 0,
    }
  },
  computed: {
    ...mapState([ 'highlighted_sample', 'data_table', 'data_features', 'matrixview', 'layout', 'primary' ]),
    ...mapGetters([ 'model_info', 'rule_info' ]),
  },
  watch: {
    layout(val) {
      if (val != null) {
        this.render()
      }
    },
    highlighted_sample(val) {
      this.render()
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
    ...mapActions([ 'tooltip', 'orderColumn', 'orderRow', 'showRepresentRules', 'showExploreRules', 'updateMatrixWidth' ]),
    onResize(){
      const width = this.$refs.matrix_parent.getBoundingClientRect().width
      this.updateMatrixWidth(width)
    },
    render() {
      const self = this
      // const min_confidence = 5
      const matrixview = this.matrixview
      const { margin, width, height } = matrixview
      const header_offset = { x: 75, y: 45 } //this.primary.has_primary_key ? 20 : 5 }

      const svg = d3.select(".matrixdiagram")
        .attr('width', width)
        .attr('height', height)

      /*<svg style="width:24px;height:24px" viewBox="0 0 24 24">
    <path fill="currentColor" d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z" />
</svg>*/
      const fixed_element = svg.selectAll('.fixed')
        .data(['fixed'])
        .enter().append('g')
        .attr('class', 'fixed')
        
      fixed_element
        .append("symbol")
        .attr("id", "markermore")
        .attr("viewBox", "0 0 24 24")
        .append("path")
        //.style("fill", "#333")
        .attr("d", "M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z")

      fixed_element
        .append("symbol")
        .attr("id", "markerexpand")
        .attr("viewBox", "0 0 24 24")
        .append("path")
        //.style("fill", "#333")
        .attr("d", "M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z")

      fixed_element
        .append("symbol")
        .attr("id", "markercollapse")
        .attr("viewBox", "0 0 24 24")
        .append("path")
        //.style("fill", "#333")
        .attr("d", "M7.41,15.41L12,10.83L16.59,15.41L18,14L12,8L6,14L7.41,15.41Z")
      
      fixed_element
        .append("text")
        .attr("class", 'title')
        .attr("dx", 10)
        .attr("dy", 30)
        .style("font-family", "Arial")
        .style("font-size", "15px")
        .style("font-weight", 500)
        .style("fill", "rgba(0,0,0,0.6)")
        .style("color", "rgba(0,0,0,0.6)")
        .text('Rules')

      const view_height = this.matrixview.height - this.matrixview.margin.bottom - this.matrixview.margin.top
      const view_width = this.matrixview.width

      const scroll = svg.select('.scrollbar_container')
      scroll.selectAll('*').remove()

      if (view_height < this.layout.height) {
        this.current_scroll = 5
        const barheight = view_height * view_height / (this.layout.height + 50)
        new Scrollbar(scroll)
          .vertical(true)
          .sliderLength(barheight)
          .position(view_width - 20, this.matrixview.margin.top, view_height)
          .onscroll((x, sx, delta) => this.current_scroll = 5 - sx / barheight * view_height)
          .attach()
      } else {
        this.current_scroll = 0
      }
    
      const header_container = svg.select(".header_container")
      const rule_canvas = svg.select(".rule_canvas")
      const status_container = svg.select(".status_container")
      
      const layout = this.layout
      
      function brushed({selection}) {
        //rule_canvas.selectAll('g.row')
        //  .select(".glyph").select("circle")
        //  .attr("fill", "darkgray")
        if (self.matrixview.is_zoomed) {
          self.showRepresentRules()
        } else {
          const selected_rules = layout.rows
            .filter(d => d.y >= selection[0] && d.y + d.height <= selection[1])
            .map(d => d.rule.id)
          self.showExploreRules(selected_rules)
        }
      }

      function brushing({selection}) {
        const selected_rules = layout.rows
          .filter(d => d.y >= selection[0] && d.y + d.height <= selection[1])
          .map(d => d.rule.id)
        const is_selected_rule = new Set(selected_rules)
        rule_canvas.selectAll('g.row')
          .select(".glyph circle")
          .attr("fill", d => is_selected_rule.has(d.rule.id) ? "#333" : "darkgray")
        rule_canvas.selectAll('g.row')
          .select(".glyph line")
          .attr("stroke", d => is_selected_rule.has(d.rule.id) ? "#333" : "darkgray")
      }
      
      const xrange = [Math.min(...layout.cols.map(d => d.x)), Math.max(...layout.cols.map(d => d.x)) + matrixview.coverage_width]
      const yrange = [Math.min(...layout.rows.map(d => d.y)), Math.max(...layout.rows.map(d => d.y + d.height))]

      function updateStatus() {
        status_container.selectAll('*').remove()
        let count_btn = status_container.append('g')
          .attr('transform', 'translate(-45,5)')
          .on('mouseover', function(){
            d3.select(this).select('rect.background').attr('fill-opacity', .8).attr('stroke-width', 1)
              .attr('stroke', 'black')
          }).on('mouseout', function(){
            d3.select(this).select('rect.background').attr('fill-opacity', .3).attr('stroke-width', .3)
              .attr('stroke', 'lightgray')
          }).on('click', function(ev, d) {
            self.orderRow()
          })

        count_btn.append('rect')
          .attr('class', 'background')
          .attr('y', 0)
          .attr('x', -1)
          .attr('width', 80)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)

        status_container.append('rect')
          .attr('class', 'background')
          .attr('y', 4)
          .attr('x', 35)
          .attr('width', layout.width - self.matrixview.margin.left)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('fill', 'none')
        
        count_btn.append('text')
          .attr('dx', 5)
          .attr('dy', 15)
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .text('cover num')

        const status_text = status_container.append('g')
          .attr('class', 'status')
          .attr('transform', 'translate(0, 70)')
          .style('user-select', 'none')

        status_text.append('text')
          .attr('dx', layout.width + self.matrixview.margin.right - self.model_info.length * 8 + 30)
          .attr('font-size', '16px')
          .attr('font-family', 'Arial')
          .text(self.model_info)

        status_text.append('text')
          .attr('dx', layout.width + self.matrixview.margin.right - self.rule_info.length * 8)
          .attr('font-size', '16px')
          .attr('font-family', 'Arial')
          .attr('dy', 16)
          .text(self.rule_info)

        const status_orders = status_container.select('g.order')
          .data(self.matrixview.order_keys).enter()
          .append('g')
          .attr('class', 'order')
          .attr('transform', (d, i) => `translate(${135 + i * 110}, 60)`)

        status_orders.append('rect')
          .attr('width', 100)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)
      }

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
        
        let col = header_container.selectAll('g.col')
          .data(cols_data, d => d.index)

        let col_join = col.enter().append('g')
          .attr('class', 'col')
        col.exit().selectAll('*').remove()

        col = header_container.selectAll('g.col')
          .data(cols_data, d => d.index)
          //.call(drag)

        col_join.append('path').attr('class', 'header')
        col_join.append('text').attr('class', 'label')
        col_join.append('g').attr('class', 'axis')
        col_join.append('text').attr('class', 'count')

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
            for (let i = 0; i < self.matrixview.order_keys.length; ++i) {
              if (self.matrixview.order_keys[i].key == d.index) {
                prefix = self.matrixview.order_keys[i].order == 1 ? '▲' : '▼'
              }
            }
            let name = d.name
            if (d.name.length > 20) {
              for (let i = 17; i < name.length; ++i) {
                if (name[i] == ' ') {
                  name = name.slice(0, i) + ' ...'
                  break
                }
              }
            }
            return prefix + name
          })
          .style('user-select', 'none')

        col.select('rect.background')
          .attr('width', d => d.width)
          .attr('height', d => d.height)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', matrixview.cell_stroke_width)
          .attr('fill', 'white')

        col.select('text.count')
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .attr('dx', d => 5)
          .attr('dy', d => d.height + 20)
          .text(d => d.count > 0 ? d.count : '')
        
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
          //.transition().duration(matrixview.duration)
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
          .style('opacity', 0).remove()
        
        let row_join = row.enter().append('g')
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .attr('class', 'row')
          .style('opacity', 0)

        row_join.selectAll(".glyph")
          .data(d => [d]).enter()
          .append('g')
          .attr('class', 'glyph')
          .attr('transform', `translate(2.5, 0)`)
          
        row = row.merge(row_join)
        row.select('.glyph').selectAll('*').remove()
        let represent_glyph = row.filter(d => d.rule.represent)
          .select('.glyph')
          //.attr("opacity", .5)

        represent_glyph.selectAll('*').remove()
        
        represent_glyph
          .append('line')
          .attr('class', 'extend')
          .attr('x1', 0)
          .attr('x2', 90)
          .attr('y1', d => d.glyphheight / 2)
          .attr('y2', d => d.glyphheight / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')

        represent_glyph
          .append('circle')
          .attr('class', 'extend')
          .attr('cx', 90)
          .attr('cy', d => d.glyphheight / 2)
          .attr('r', d => d.glyphheight / 2 - 2)
          .attr('fill', 'darkgray')
          .attr('stroke', 'none')

        let rep_glyph = represent_glyph
          .append('g')
          .attr('class', 'represent_glyph')
          .on('mousemove', function(ev, d) {
            self.tooltip({ type: "text", data: `represent ${Number(d.attr.num_children)} rules`})
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
          })
          .on('mouseover', function(){
            self.tooltip({ type: "show" })
            d3.select(this).select("rect.bg")
              .transition().duration(matrixview.duration / 2)
              .attr('x', -1.5)
              .attr('y', -1)
              .attr('width', d => d.attr.num + 2)
              .attr('height', d => d.glyphheight + 3)
              .attr('stroke', "#333")
              .attr("stroke-width", '2.5px')
            d3.select(this).select(".arrow")
              .style("display", "block")
            d3.select(this.parentNode.parentNode).raise()
            /*
            d3.select(this).selectAll("rect.dot")
              .transition().duration(matrixview.duration / 2)
              .attr('width', 3)
              .attr('height', 3)
              .attr('fill', "#333")*/
            d3.select(this.parentNode).select("line.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('stroke', '#333')
              .attr('stroke-width', '3px')
            d3.select(this.parentNode).select("circle.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('r', d => d.glyphheight / 2)
              .attr('fill', '#333')
          })
          .on('mouseout', function(){
            self.tooltip({ type: "hide" })
            d3.select(this).select("rect.bg")
              .transition().duration(matrixview.duration / 2)
              .attr('x', 0)
              .attr('y', 1)
              .attr('width', d => d.attr.num)
              .attr('height', d => d.glyphheight)
              .attr('stroke', "darkgray")
              .attr("stroke-width", '1.5px')
            d3.select(this).select(".arrow")
              .style("display", self.matrixview.is_zoomed ? "block" : "none")
              /*
            d3.select(this).selectAll("rect.dot")
              .transition().duration(matrixview.duration / 2)
              .attr('width', 2)
              .attr('height', 2)
              .attr('fill', "darkgray")*/
            d3.select(this.parentNode).select("line.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('stroke', 'darkgray')
              .attr('stroke-width', '2px')
            d3.select(this.parentNode).select("circle.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('r', d => d.glyphheight / 2 - 2)
              .attr('fill', 'darkgray')
          })
          .on('click', function(ev, d){
            if (self.matrixview.is_zoomed) {
              self.showRepresentRules()
            } else {
              self.showExploreRules([d.rule.id])
            }
          })

        rep_glyph
          .append('rect')
          .attr('class', 'bg')
          .attr('x', self.matrixview.is_zoomed ? -1.5 : 0)
          .attr('y', self.matrixview.is_zoomed ? -1 : 1)
          .attr('width', d => self.matrixview.is_zoomed ? d.attr.num + 2 : d.attr.num)
          .attr('height', d => self.matrixview.is_zoomed ? d.glyphheight + 3 : d.glyphheight)
          .attr('fill', '#f7f7f7')
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '1.5px')

        rep_glyph
          .append("use")
          .attr("href", self.matrixview.is_zoomed ? "#markercollapse" : "#markerexpand")
          .attr("x", -3)
          .attr("class", "arrow")
          .attr("width", 15)
          .attr("height", 15)
          .attr("y", -3)
          .style("display", self.matrixview.is_zoomed ? "block" : "none")
          .style("fill", "#333")
/*
        let represent_glyph_dot = rep_glyph
          .append("g")
          .attr("class", "dot")

        represent_glyph_dot
          .append("rect")
          .attr("x", 12)
          .attr("y", d => d.height / 2 - 1)
          .attr("height", 2)
          .attr("width", 2)
          .attr("class", "dot")
          .attr("fill", "darkgray")

        represent_glyph_dot
          .append("rect")
          .attr("x", 16)
          .attr("y", d => d.height / 2 - 1)
          .attr("height", 2)
          .attr("width", 2)
          .attr("class", "dot")
          .attr("fill", "darkgray")

        represent_glyph_dot
          .append("rect")
          .attr("x", 20)
          .attr("y", d => d.height / 2 - 1)
          .attr("height", 2)
          .attr("width", 2)
          .attr("class", "dot")
          .attr("fill", "darkgray")
*/
        let nonrepresent_glyph = row.filter(d => !d.rule.represent)
          .select('.glyph')
          //.attr("opacity", .5)

        nonrepresent_glyph
          .append('line')
          .attr('x1', 50)
          .attr('x2', 90)
          .attr('y1', d => d.glyphheight / 2)
          .attr('y2', d => d.glyphheight / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')

        nonrepresent_glyph
          .append('circle')
          .attr('cx', 90)
          .attr('cy', d => d.glyphheight / 2)
          .attr('r', d => d.glyphheight / 2 - 2)
          .attr('fill', 'darkgray')
          .attr('stroke', 'none')

        nonrepresent_glyph
          .append('line')
          .attr('x1', 50)
          .attr('x2', 50)
          .attr('y1', d => d.glyphheight / 2 - d.lastheight)
          .attr('y2', d => d.glyphheight / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')
        
        row_join = row_join.merge(row)
        
        const brush = d3.brush()
          .on("end", brushed)

        canvas_brush
          .call(
            d3.brushY().extent([[xrange[0] - 60, yrange[0]], [xrange[0], yrange[1]]])
            .on("brush", brushing)
            .on("end", brushed)
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
        cell.append('rect').attr('class', 'barbg')
        cell.append('g').attr('class', 'bargroup')

        row = row.merge(row_join)
        cell = row.selectAll('g.cell')
          .on("mouseover", function(){
            self.tooltip({ type: "show" })
          })
          .on("mouseout", function(){
            self.tooltip({ type: "hide" })
          })
          .on("mousemove", function(ev, d){
            if (d.feature.type == 'categoric') {
              const s = d.cond.range.reduce((a, b) => a + b)
              let text = `${d.name}: `
              let items = []
              if (s <= d.cond.range.length / 2) {
                for (let i = 0; i < d.cond.range.length; ++i) {
                  if (d.cond.range[i]) {
                    items.push(d.feature.values[i])
                  }
                }
              } else {
                text += 'NOT '
                for (let i = 0; i < d.cond.range.length; ++i) {
                  if (!d.cond.range[i]) {
                    items.push(d.feature.values[i])
                  }
                }
              }
              //console.log(text, items, d.cond.range, d.feature)
              text += items.join(', ')
              self.tooltip({
                type: "text",
                data: text,
              })
            } else {
              self.tooltip({
                type: "text",
                data: `${Number(Math.max(d.feature.range[0], d.cond.range[0])).toFixed(3)} <= ${d.name} < ${Number(Math.min(d.feature.range[1], d.cond.range[1])).toFixed(3)}`
              })
            }
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
          })
          .raise()

        row_extend = row.selectAll('g.extended')

        cell.select('rect.barbg')
          .attr('x', d => d.x)
          .attr('width', d => d.width)
          .attr('height', d => d.height)
          .attr('fill', d => 
            self.matrixview.is_zoomed && d.represent ? 
              d3.interpolateLab('white', d.fill)(0.05) :
              d3.interpolateLab('white', d.fill)(0.25)
            )
          .attr('stroke', d => 
            self.matrixview.is_zoomed && d.represent ? '#555' : 'none'
          )
          .attr('stroke-width', '1px')
          .attr('opacity', 1)

        canvas_brush.raise()

        cell.select(".histogram").remove()

        cell.select('g.bargroup')
          
        const cell_bars = cell.select('g.bargroup')

        cell_bars
          //.attr("opacity", 0)
          .transition().duration(matrixview.duration)
          .attr('transform', d => `translate(${d.x},${0})`)
          //.attr("opacity", 1)

        cell_bars.selectAll('rect.bar')
          .data(d => d.elements).exit().remove()
          
        cell_bars.selectAll('rect.bar')
          .data(d => d.elements).enter()
          .append('rect').attr('class', 'bar')

        cell_bars.selectAll('rect.bar')
          .data(d => d.elements)
          .attr('x', d => d.x0 + 0.75)
          .attr('y', 1)
          .attr('width', d => d.x1 - d.x0 - 1.5)
          .attr('height', d => d.h - 1.5)
          .attr('fill', d => {
              if (self.matrixview.is_zoomed && d.represent) {
                return d3.interpolateLab('#ccc', d.fill)(0.2)
              } else if (!d.neg) {
                return d3.interpolateLab('white', d.fill)(1)
              } else {
                return 'none'
              }
          })
          .attr('stroke', d => {
              if (self.matrixview.is_zoomed && d.represent) {
                return d3.interpolateLab('#ccc', d.fill)(0.2)
              } else {
                return d3.interpolateLab('white', d.fill)(1)
              }
          })
          .attr('stroke-width', '1.5px')

        cell_bars.selectAll('line.bar')
          .data(d => d.elements)
          .filter(d => !(self.matrixview.is_zoomed && d.represent) && d.neg)
          .exit().remove()

        cell_bars.selectAll('line.bar')
          .data(d => d.elements)
          .enter()
          .append('line').attr('class', 'bar')

        cell_bars.selectAll('line.bar')
          .data(d => d.elements)
          .filter(d => !(self.matrixview.is_zoomed && d.represent) && d.neg)
          .attr('x1', d => d.x0)
          .attr('y1', d => d.h - 1)
          .attr('x2', d => d.x1)
          .attr('y2', d => 1)
          .attr('stroke', d => d3.interpolateLab('white', d.fill)(1))
          .attr('stroke-width', '1.5px')

        const chart_cell = cell.filter(d => self.matrixview.is_zoomed && d.represent)
        chart_cell.each(function(d){
          const data = []
          for (let i of d.samples) {
            data.push(self.data_table[i])
          }

          const chart = HistogramChart()
            .data(data)
            .valueTicks(d.feature.type == "categoric" ? d.feature.values : d.feature.range)
            .x(d.name)
            .width(d.width)
            .datatype(d.feature.type == "categoric" ? "category" : "number")
            .height(d.height)
            .color(d.fill)

          d3.select(this)
            .append("g")
            .attr("class", "histogram")
            .attr("transform", `translate(${d.x}, 0)`)
            .attr("opacity", 0)
            .call(chart)

          d3.select(this)
            .select("g.histogram")
            .transition().duration(matrixview.duration)
            .delay(matrixview.duration)
            .attr("opacity", 1)
        })

        row_extend.select('rect.bar')
          .on("mouseover", function(){
            self.tooltip({ type: "show" })
          })
          .on("mouseout", function(){
            self.tooltip({ type: "hide" })
          })
          .on("mousemove", function(ev, d){
            self.tooltip({
              type: "text",
              data: `${Number(d.value).toFixed(3)}`
            })
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
          })
          .raise()

        row_extend.select('rect.bar')
          .attr('x', d => d.x1)
          .attr('width', d => d.x2 - d.x1)
          .attr('height', d => d.height)
          .attr('fill', d => d.fill)
        
        row
          .transition().duration(matrixview.duration)
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .transition().duration(matrixview.duration)
          .style('opacity', d => self.highlighted_sample == null ? 1 : d.samples.has(self.highlighted_sample) ? 1 : 0.3)
//          .style('opacity', 1)//d => d.rule.represent ? 1 : 0.5)

      }
      
      function update() {
        updateCols()
        updateRows()
        updateStatus()
      }
      
      update()
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
</style>
