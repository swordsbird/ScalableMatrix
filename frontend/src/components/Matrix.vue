<template>
  <div ref="matrix_parent" :style="`position: absolute; ${positioning}`" v-resize="onResize">
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
    <svg class="matrixdiagram" style="width: 100%; height: 100%">
      <clipPath id="rule_clip">
        <rect :width="`${matrixview.width + 5}`" 
          :height="`${matrixview.height - matrixview.margin.bottom - matrixview.margin.top + 5}`">
        </rect>
      </clipPath>
      <g class="rule_canvas_container"
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width},${matrixview.margin.top})`">
        <g class="rule_outer" clip-path="url(#rule_clip)">
          <g class="rule_canvas" :transform="`translate(0, ${current_scroll})`">
          </g>
        </g>
      </g>
      <g class="header_container" 
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width},${matrixview.margin.top})`">
      </g>
      <g class="status_container" 
        :transform="`translate(${matrixview.margin.left}, ${matrixview.height - matrixview.margin.bottom})`">
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
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    }
  },
  data() {
    return {
      current_col: null,
      current_row: null,
      current_scroll: 0,
    }
  },
  computed: {
    ...mapState([ 'highlighted_sample', 'data_table', 'data_features', 'matrixview', 'featureview', 'layout', 'primary' ]),
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
  // beforeDestroy () {
  //   if (typeof window === 'undefined') return
  //   window.removeEventListener('resize', this.onResize, { passive: true })
  // },
  // async mounted() {
  //   window.addEventListener('resize', this.onResize, { passive: true })
  //   this.onResize()
  // },
  methods: {
    ...mapActions([ 'tooltip', 'orderColumn', 'orderRow', 'showRules', 'updateMatrixSize' ]),
    onResize(){
      const width = this.$refs.matrix_parent.getBoundingClientRect().width
      const height = this.$refs.matrix_parent.getBoundingClientRect().height
      this.updateMatrixSize({ width, height })
    },
    render() {
      const self = this
      // const min_confidence = 5
      const matrixview = this.matrixview
      const header_offset = { x: 100, y: 8 } //this.primary.has_primary_key ? 20 : 5 }

      const svg = d3.select(".matrixdiagram")
        .attr('width', this.matrixview.width)
        .attr('height', this.matrixview.height)

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

      if (0 && view_height < this.layout.height) {
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
        /*
        if (self.matrixview.zoom_level > 0) {
          self.showRepresentRules()
        } else {
        */
        const selected_rules = layout.rows
          .filter(d => d.y >= selection[0] && d.y + d.height <= selection[1])
          .map(d => d.rule.id)
        self.showRules(selected_rules)
        //}
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
          .attr('x', -20)
          .attr('width', 98)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)

        status_container.append('rect')
          .attr('class', 'background')
          .attr('y', 4)
          .attr('x', 35)
          .attr('width', layout.width - self.matrixview.margin.left - 15)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('fill', 'none')

        count_btn.append('path')
          .attr('d', "M41 288h238c21.4 0 32.1 25.9 17 41L177 448c-9.4 9.4-24.6 9.4-33.9 0L24 329c-15.1-15.1-4.4-41 17-41zm255-105L177 64c-9.4-9.4-24.6-9.4-33.9 0L24 183c-15.1 15.1-4.4 41 17 41h238c21.4 0 32.1-25.9 17-41z")
          .attr('transform', "translate(74.5, 4) scale(0.04) rotate(90)")
          .attr("opacity", matrixview.sort_by_cover_num ? .8 : .2)
        
        count_btn.append('text')
          .attr('dx', -15)
          .attr('dy', 14)
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .text('cover num')

        // const status_text = status_container.append('g')
        //   .attr('class', 'status')
        //   .attr('transform', 'translate(0, 70)')
        //   .style('user-select', 'none')

        // status_text.append('text')
        //   .attr('dx', layout.width + self.matrixview.margin.right - self.model_info.length * 8 + 30)
        //   .attr('font-size', '16px')
        //   .attr('font-family', 'Arial')
        //   .text(self.model_info)

        // status_text.append('text')
        //   .attr('dx', layout.width + self.matrixview.margin.right - self.rule_info.length * 8)
        //   .attr('font-size', '16px')
        //   .attr('font-family', 'Arial')
        //   .attr('dy', 16)
        //   .text(self.rule_info)

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

        col.exit().selectAll('*').remove()

        let col_join = col.enter().append('g')
          .attr('class', 'col')
          .attr('transform', d => `translate(${d.x},${d.y})`)

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
        
        function headerInteraction(x) {
          x.on('mouseover', function(){
            d3.select(this)
              .select('rect.background')
              .attr('stroke', matrixview.cell.highlight_stroke_color)
              .attr('stroke-width', matrixview.cell.highlight_stroke_width)
            d3.select(this)
              .select('path.header')
              .attr('fill-opacity', matrixview.cell.highlight_header_opacity)
              .attr('stroke', matrixview.cell.highlight_stroke_color)
              .attr('stroke-width', matrixview.cell.highlight_stroke_width)
          }).on('mouseout', function(){
            d3.select(this)
              .select('rect.background')
              .attr('stroke', matrixview.cell.stroke_color)
              .attr('stroke-width', matrixview.cell.stroke_width)
            d3.select(this)
              .select('path.header')
              .attr('fill-opacity', matrixview.cell.header_opacity)
              .attr('stroke', matrixview.cell.stroke_color)
              .attr('stroke-width', matrixview.cell.stroke_width)
          }).on('click', function(ev, d) {
            self.orderColumn(d.index)
          })
        }

        col.select('path.header')
          .attr('fill', 'lightgray')
          .attr('fill-opacity', matrixview.cell.header_opacity)
          .attr('stroke', matrixview.cell.stroke_color)
          .attr('stroke-width', matrixview.cell.stroke_width)

        col.select('path.header')
          .transition().duration(matrixview.duration)
          .attr('d', d => {
            return `M0,0 L${d.width},0 L${d.width},${-header_offset.y} L${d.width+header_offset.x*1.5},${-header_offset.y-header_offset.x}
                         L${header_offset.x*1.5},${-header_offset.y-header_offset.x} L${0},${-header_offset.y} z`
          })

        col.selectAll('g.hint').remove()
        const hint_glyph = col.filter(d => d.hint_change)
          .append('g')
          .attr('class', 'hint')
          .attr('transform', `translate(${header_offset.x*1.5},${-header_offset.y-header_offset.x})`)
        
        hint_glyph.append('text')
          .attr('dx', 4)
          .attr('dy', -2)
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .attr('fill', 'green')
          .text(d => `▲ +${d.delta}`)

        col.selectAll('.category').remove()

        col.filter(d => d.show_axis && d.type == 'categoric')
          .each(function(d){
            const width = d.width / d.values.length
            const offset_x = header_offset.x - 20
            const offset_y = header_offset.y
            let category = d3.select(this)
              .selectAll('.category')
              .data(d.values).enter()
              .append('g')
              .attr('class', 'category')
              .attr('transform', (e, i) => `translate(${width * i}, 0)`)
            
            category
              .append('path')
              .attr('fill', 'lightgray')
              .attr('fill-opacity', matrixview.cell.header_opacity)
              .attr('stroke', matrixview.cell.stroke_color)
              .attr('stroke-width', matrixview.cell.stroke_width)
              .attr('d', d => {
                return `M0,0 L${width},0 L${width},${-offset_y} L${width+offset_x*1.5},${-offset_y-offset_x}
                            L${offset_x*1.5},${-offset_y-offset_x} L${0},${-offset_y} z`
              })
            
            const dy = width / 2 - header_offset.y + 1.5
            const len = Math.min(matrixview.maxlen, Math.max(...d.values.map(e => e.length)))
            category
              .append('text')
              .attr('transform', `rotate(-35)`)
              .attr('font-size', '12px')
              .attr('font-family', 'Arial')
              .attr('dx', dy + 28 + (matrixview.maxlen - len) / 2 * 3)
              .attr('dy', dy)// header_offset.y)
              .text(name => name.length < matrixview.maxlen ? name : name.slice(0, matrixview.maxlen) + "...")
              //.style('user-select', 'none')
          })

        col.call(headerInteraction)
        
        col.select('text.label')
          .attr('transform', d => d.show_axis ? '' : `rotate(-35)`)
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .attr('dx', d => {
            if (!d.show_axis) {
              return 30
            } else if (d.type == 'categoric') {
              return 40 + header_offset.x
            } else {
              return 40
            }
          })
          .attr('dy', d => {
            if (!d.show_axis) {
              return 18 - header_offset.y
            } else if (d.type == 'categoric') {
              return -header_offset.x + 3
            } else {
              return -header_offset.y - 13
            }
          })
          .text(d => {
            let prefix = ''
            for (let i = 0; i < self.matrixview.order_keys.length; ++i) {
              if (self.matrixview.order_keys[i].key == d.index) {
                prefix = self.matrixview.order_keys[i].order == 1 ? '▲' : '▼'
              }
            }
            const maxlen = d.show_axis ? matrixview.maxlen + 5 : matrixview.maxlen
            let name = d.name.length < maxlen ? d.name : d.name.slice(0, maxlen) + "..."
            return prefix + name
          })
          .style('user-select', 'element')

        col.select('rect.background')
          .attr('stroke', matrixview.cell.stroke_color)
          .attr('stroke-width', matrixview.cell.stroke_width)
          .attr('height', d => d.height)
          .attr('fill', 'none')

        col.select('rect.background')
          .transition().duration(matrixview.duration)
          .attr('width', d => d.width)

        col.select('text.count')
          .attr('font-size', '14px')
          .attr('font-family', 'Arial')
          .attr('dx', d => 5)
          .attr('dy', d => d.height + 20)
          .text(d => d.count > 0 ? d.count : '')
        
        col.filter(d => !d.show_axis && d.type == 'numeric').each(function(d){
          d3.select(this).select('g.axis').selectAll('*').remove()
        })
        col.filter(d => d.show_axis && d.type == 'numeric').each(function(d){
          d3.select(this).select('g.axis')
            //.attr('transform', `translate(0,${-header_offset.y + 5})`)
            .call(d3.axisTop(d.scale).ticks(4))
          d3.select(this).select('g.axis').raise()
        })
        
        col
          .transition().duration(matrixview.duration)
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
        let represent_glyph = row//.filter(d => d.rule.represent)
          .select('.glyph')
          //.attr("opacity", .5)

        represent_glyph.selectAll('*').remove()
        
        represent_glyph
          .append('line')
          .attr('class', 'extend')
          .attr('x1', d => d.rule.represent ? 0 : 30)
          .attr('x2', 90)
          .attr('y1', d => d.glyphheight / 2 + 1)
          .attr('y2', d => d.glyphheight / 2 + 1)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')

        represent_glyph
          .append('circle')
          .attr('class', 'extend')
          .attr('cx', 90)
          .attr('cy', d => d.glyphheight / 2 + 1)
          .attr('r', d => d.glyphheight / 2 - 0.5)
          .attr('fill', 'darkgray')
          .attr('stroke', 'none')

        let rep_glyph = represent_glyph
          .append('g')
          .attr('class', 'represent_glyph')
          .on('mousemove', function(ev, d) {
            if (d.attr.num_children == 0) return
            self.tooltip({ type: "text", data: `represent ${Number(d.attr.num_children)} rules`})
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
          })
          .on('mouseover', function(ev, d){
            if (d.attr.num_children == 0) return
            self.tooltip({ type: "show" })
            d3.select(this).select("rect.bg")
              .transition().duration(matrixview.duration / 2)
              .attr('x', d => (d.rule.represent ? 0 : 30) - 1)
              .attr('y', 1 - 1.5)
              .attr('width', d => (d.rule.represent ? d.attr.num : (d.attr.num * 0.7)) + 2)
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
          .on('mouseout', function(ev, d){
            if (d.attr.num_children == 0) return
            self.tooltip({ type: "hide" })
            d3.select(this).select("rect.bg")
              .transition().duration(matrixview.duration / 2)
              .attr('x', d => d.rule.represent ? 0 : 30)
              .attr('y', 1)
              .attr('width', d => d.rule.represent ? d.attr.num : (d.attr.num * 0.7))
              .attr('height', d => d.glyphheight)
              .attr('stroke', "darkgray")
              .attr("stroke-width", '1.5px')
            d3.select(this).select(".arrow")
              .style("display", d => (self.matrixview.zoom_level > 0 && d.rule.level < self.matrixview.zoom_level) ? "block" : "none")
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
              .attr('r', d => d.glyphheight / 2 - 0.5)
              .attr('fill', 'darkgray')
          })
          .on('click', function(ev, d){
            console.log(d)
            if (self.matrixview.zoom_level > d.rule.level) {
              self.showRules(null)
            } else {
              self.showRules([d.rule.id])
            }
          })

        rep_glyph
          .append('rect')
          .attr('class', 'bg')
          .attr('x', d => d.rule.represent ? 0 : 30)
          .attr('y', 1)
          .attr('width', d => d.rule.represent ? d.attr.num : (d.attr.num * 0.7))
          .attr('height', d => d.glyphheight)
          .attr('fill', '#f7f7f7')
          .attr('stroke', 'darkgray')
          .style("display", d => (d.attr.num_children > 0 || d.rule.level == 0) ? "block" : "none")
          .attr('stroke-width', '1.5px')

        rep_glyph
          .append("use")
          .attr("href", d => self.matrixview.zoom_level == d.rule.level ? "#markerexpand" : "#markercollapse")
          .attr("x",  d => d.rule.represent ? -3 : 27)
          .attr("class", "arrow")
          .attr("width", 15)
          .attr("height", 15)
          .attr("y", -3)
          .style("display", d => (self.matrixview.zoom_level > 0 && d.rule.level < self.matrixview.zoom_level) ? "block" : "none")
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
/*
        nonrepresent_glyph
          .append('line')
          .attr('x1', 30)
          .attr('x2', 90)
          .attr('y1', d => d.glyphheight / 2)
          .attr('y2', d => d.glyphheight / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')

        nonrepresent_glyph
          .append('circle')
          .attr('cx', 90)
          .attr('cy', d => d.glyphheight / 2)
          .attr('r', d => d.glyphheight / 2 - 0.5)
          .attr('fill', 'darkgray')
          .attr('stroke', 'none')
*/
        nonrepresent_glyph
          .append('line')
          .attr('x1', 30)
          .attr('x2', 30)
          .attr('y1', d => d.glyphheight / 2 - d.lastheight)
          .attr('y2', d => d.glyphheight / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')
          .attr('class', 'parent')

        nonrepresent_glyph
          .select('line.parent')
          .attr('opacity', 0)

        nonrepresent_glyph
          .select('line.parent')
          .transition().duration(matrixview.duration)
          .delay(matrixview.duration)
          .attr('opacity', 1)

/*
        nonrepresent_glyph
          .append('rect')
          .attr('class', 'bg')
          .attr('fill', '#f7f7f7')
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '1.5px')
*/
        row_join = row_join.merge(row)
        
        const brush = d3.brush()
          .on("end", brushed)

        canvas_brush
          .call(
            d3.brushY().extent([[xrange[0] - 30, yrange[0]], [xrange[0], yrange[1]]])
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
                data: `${Number(Math.max(d.feature.range[0], d.cond.range[0])).toFixed(3)} < ${d.name} <= ${Number(Math.min(d.feature.range[1], d.cond.range[1])).toFixed(3)}`
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
          .attr('fill', d => d.show_hist ? 
              d3.interpolateLab('white', d.fill)(0.05) :
              d3.interpolateLab('white', d.fill)(0.25)
            )
          .attr('stroke', d => d.show_hist ? '#555' : 'none')
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
          .attr('width', d => Math.max(d.x1 - d.x0 - 1.5, 0))
          .attr('height', d => Math.max(d.h - 1.5, 0))
          .attr('fill', d => {
              if (d.show_hist) {
                return d3.interpolateLab('#ccc', d.fill)(0.2)
              } else if (!d.neg) {
                return d3.interpolateLab('white', d.fill)(1)
              } else {
                return 'white'
              }
          })
          .attr('stroke', d => {
              if (d.show_hist) {
                return d3.interpolateLab('#ccc', d.fill)(0.2)
              } else {
                return d3.interpolateLab('white', d.fill)(1)
              }
          })
          .attr('stroke-width', '1.5px')
        /*
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
        */
        const chart_cell = cell.filter(d => d.show_hist)
        chart_cell.each(function(d){
          const data = []
          //console.log(d.samples, d.feature)
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
          .style('opacity', d => self.highlighted_sample === undefined ? 1 : d.samples.has(self.highlighted_sample) ? 1 : 0.3)
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
