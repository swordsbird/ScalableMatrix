<template>
  <div
    ref="container" class="white"
    :style="`position: absolute; ${positioning}; user-select: none`"
    v-resize="onResize"
  ><svg ref="tableview" style="width: 100%; height: 100%"></svg> 
  </div>
</template>

<script>
import * as d3 from 'd3'
import { mapActions, mapGetters, mapState } from 'vuex'
import SVGTable from '../libs/svgtable'
export default {
  name: 'DataTable',
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    },
    display: Boolean
  },
  computed: {
    ...mapGetters(['filtered_data']),
    ...mapState(['highlighted_sample', 'crossfilter', 'data_shaps', 'data_table'])
  },
  watch: {
    crossfilter () { this.renderTable() },
    highlighted_sample () { this.renderTable() }
  },
  methods: {
    ...mapActions(['highlightSample']),
    renderTable () {
      const width = this.$refs.container.getBoundingClientRect().width - 4
      const height = this.$refs.container.getBoundingClientRect().height - 4

      const svg = d3.select(this.$refs.tableview)
      svg.selectAll('*').remove()

      if (!this.display) return

      const reordered_data = this.highlighted_sample ? (
        this.filtered_data
          .filter(d => this.highlighted_sample === d._id)
          .concat(this.filtered_data.filter(d => this.highlighted_sample !== d._id))
      ) : this.filtered_data

      // console.log(reordered_data[0])
      // console.log(this.filtered_data)
      // console.log(this.data_shaps)
      // console.log(this.data_table)

      const self = this
      new SVGTable(svg)
        .size([width, height])
        .fixedRows(this.highlighted_sample !== undefined ? 1 : 0)
        .fixedColumns(1)
        .rowsPerPage(25)
        .defaultNumberFormat(',.0d')
        .style({ border: false })
        .cellRender((rect, fill, isHeader, isFixed) => {
          if (isHeader) return false
          console.log(rect)
        })
        .highlightRender(r => {
          r.attr('fill', d => d ? '#300' : '#FFF')
        })
        .data(reordered_data)
        .onclick(function (ctx, cell) {
          console.log(cell)
          let sample_id = this._data[cell.rowIndex]._id
          self.highlightSample(sample_id)
        })
        .render()
    },
    onResize () {
      this.renderTable()
    }
  }
}
</script>

<style>

</style>