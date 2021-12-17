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
import { mapGetters, mapState } from 'vuex'
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
    ...mapState(['highlighted_sample', 'crossfilter'])
  },
  watch: {
    crossfilter () { this.renderTable() },
    highlighted_sample () { this.renderTable() }
  },
  methods: {
    renderTable () {
      const width = this.$refs.container.getBoundingClientRect().width - 4
      const height = this.$refs.container.getBoundingClientRect().height - 4

      console.log(width, height)

      const svg = d3.select(this.$refs.tableview)
      svg.selectAll('*').remove()

      const reordered_data = this.highlighted_sample ? (
        this.filtered_data
          .filter(d => this.highlighted_sample === d._id)
          .concat(this.filtered_data.filter(d => this.highlighted_sample !=- d._id))
      ) : this.filtered_data

      new SVGTable(svg)
        .size([width, height])
        .fixedRows(this.highlighted_sample ? 1 : 0)
        .fixedColumns(1)
        .rowsPerPage(25)
        .defaultNumberFormat(',.0d')
        .style({ border: false })
        .data(reordered_data)
        .onclick((ctx, cell) => {
          console.log(ctx)
          const sample_id = this.filtered_data[cell.rowIndex]._id
          console.log(sample_id)
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