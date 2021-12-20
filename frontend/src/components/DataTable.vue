<template>
  <div
    ref="container" class="white"
    :style="`position: absolute; ${positioning}; user-select: none`"
    v-resize="onResize"
  >
    <v-btn
      style="position: absolute; top: 0px; right: 160px"
      :color="showShaps ? 'primary': 'grey'" text small
      @click="toggleShowShaps"
      >
      shap
    </v-btn>
    <svg ref="tableview" style="width: 100%; height: 100%"></svg> 
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
  data: () => ({
    showShaps: false,
    instance: null,
  }),
  computed: {
    ...mapGetters(['filtered_data']),
    ...mapState(['highlighted_sample', 'crossfilter', 'data_shaps', 'data_table', 'colorSchema'])
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

      function getWeightedColor (baseColor, shap) {
        return d3.interpolateLab('white', baseColor)(Math.sqrt(Math.sqrt(Math.abs(shap))))
      }

      const self = this
      this.instance = new SVGTable(svg)
        .size([width, height])
        .fixedRows(this.highlighted_sample !== undefined ? 1 : 0)
        .fixedColumns(1)
        .rowsPerPage(25)
        .defaultNumberFormat(',.0d')
        .style({ border: false })
        .cellRender(function (rect, fill, isHeader, isFixedRow, isFixedCol) {
          if (isHeader) return false
          if (isFixedCol) return false
          if (!self.showShaps) return false
          rect.attr('fill', d => {
            if (d.colIndex < 2) return 'white'
            else if (d.colIndex == 2) {
              return self.colorSchema[d.value == 'Yes' ? 0 : 1]
            }
            const shapVal = self.data_shaps[d.colIndex - 3][this._data[d.rowIndex]._id]
            if (shapVal !== undefined) {
              const baseColor = self.colorSchema[shapVal > 0 ? 0: 1]
              const color = getWeightedColor(baseColor, shapVal)
              if (isFixedRow) {
                // TODO: set style for the fixed row
              }
              return color
            }
            return 'white'
          })
          return true
        })
        .highlightRender(function (r) {
          r.attr('fill', item => {
            const { d, hl } = item
            if (self.showShaps) {
              if (d.colIndex < 2) return 'white'
              else if (d.colIndex == 2) {
                return self.colorSchema[d.value == 'Yes' ? 0 : 1]
              }
              const shapVal = self.data_shaps[d.colIndex - 3][this._data[d.rowIndex]._id]
              let color = 'white'
              if (shapVal !== undefined) {
                const baseColor = self.colorSchema[shapVal > 0 ? 0: 1]
                color = getWeightedColor(baseColor, shapVal)
              }
              if (hl) {
                color = d3.interpolateLab('grey', color)(0.8)
              }
              return color
            } else {
              return hl ? d3.interpolateLab('grey', 'white')(0.8) : 'white'
            }
          })
        })
        .data(reordered_data)
        .onclick(function (ctx, cell) {
          let sample_id = this._data[cell.rowIndex]._id
          self.highlightSample(sample_id)
        })
        .render()
    },
    toggleShowShaps () {
      this.showShaps = !this.showShaps
      if (this.instance) {
        this.instance.refresh()
      }
    },
    onResize () {
      this.renderTable()
    }
  }
}
</script>

<style>

</style>