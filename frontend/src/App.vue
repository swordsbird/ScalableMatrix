<template>
  <v-app>
    <v-system-bar app height="48">
      <div class="d-flex align-center text-h5 font-weight-medium">
        ExTreeEnsemble
      </div>

      <v-spacer></v-spacer>
    </v-system-bar>

    <v-main>
      <v-container fluid>
        <v-row>
          <v-col
            cols="12"
            md="2"
            class="pa-1"
          >
            <v-card class="mx-auto" id="feature_view" :max-height="topview_height">
              <Feature :render="initilized"/>
              <!--v-img src="./features.png"></v-img-->
            </v-card>
          </v-col>
          <v-col
            cols="12"
            md="8"
            class="pa-1"
          >
            <v-card
              class="mx-auto" :max-height="topview_height"
            >
              <Matrix/>
            </v-card>
          </v-col>
          <v-col
            cols="12"
            md="2"
            class="pa-1"
          >
            <v-card
              class="mx-auto" :height="topview_height"
            >
              <Info :render="initilized"/>
            </v-card>
          </v-col>
        </v-row>
        <v-row>
          <v-col class="pt-1"
            cols="12"
            md="12"
          >
            <v-card flat>
              <svg ref="tableview"></svg>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
      <div class="svg-tooltip"
      :style="{
        left: `${Math.min(page_width - tooltipview.width, tooltipview.x + 10)}px`,
        top: `${tooltipview.y - 10}px`,
        'max-width': `${tooltipview.width}px`,
        visibility: tooltipview.visibility
      }">{{ tooltipview.content }}
      </div>
    </v-main>
  </v-app>
</template>

<script>
import Matrix from './components/Matrix.vue'
import Feature from './components/Feature.vue'
import Info from './components/Info.vue'
import { mapActions, mapGetters, mapState } from "vuex"
import SVGTable from './libs/svgtable'
import * as d3 from 'd3'


export default {
  name: 'App',
  components: {
    Matrix,
    Feature,
    Info
  },
  computed: {
    ...mapGetters(['topview_height', 'filtered_data']),
    ...mapState(['tooltipview', 'highlighted_sample', 'data_table', 'crossfilter', 'data_header', 'page_width', 'matrixview'])
  },
  watch: {
    crossfilter(val) {
      this.renderDataTable()
    },
    highlighted_sample(val) {
      this.renderDataTable()
    }
  },
  methods: {
    ...mapActions(['highlightSample', 'fetchRawdata', 'updateMatrixLayout', 'updatePageSize', 'setReady']),
    onResize(){
      const width = document.documentElement.clientWidth
      const height = document.documentElement.clientHeight
      this.updatePageSize({ width, height })
      this.renderDataTable()
    },
    renderDataTable() {
      const width = this.$refs.tableview.parentNode.getBoundingClientRect().width
      const height = 250

      const svg = d3.select(this.$refs.tableview)
          .attr("width", width)
          .attr("height", height);

      svg.selectAll('*').remove()

      const reordered_data = 
        this.highlighted_sample ? (
          this.filtered_data.filter(d => this.highlighted_sample == d._id).concat(
            this.filtered_data.filter(d => this.highlighted_sample != d._id)
          )
        ) : this.filtered_data

      new SVGTable(svg)
          .size([width, height])
          .fixedRows(this.highlighted_sample ? 1 : 0)
          .fixedColumns(1)
          .rowsPerPage(25)
          .defaultNumberFormat(",.0d")
          .style({ border: false })
          .data(reordered_data)  
          .onclick((ctx, cell) => {
            const sample_id = this.filtered_data[cell.rowIndex]._id
            this.highlightSample(sample_id)
          })
          //.onhighlight((ctx, d) => { console.log(ctx, ) })
          .render();
    }
  },
  data: () => {
    return {
      drawer: null,
      tab: 'Data Table',
      initilized: false,
    }
  },
  beforeDestroy () {
    if (typeof window === 'undefined') return
    window.removeEventListener('resize', this.onResize, { passive: true })
  },
  async mounted() {
    await this.fetchRawdata()
    await this.setReady()
    await this.updateMatrixLayout()
    window.addEventListener('resize', this.onResize, { passive: true })
    this.onResize()
    this.renderDataTable()
    this.initilized = true
  }
}
/*
离散值 留出空隙 不要连续
最下面加一条横线

Not 空心 / Yes 实心
sync with MingYao

crossfilter
规则filter，有个明确的confirm button
histogram debug
*/
</script>

<style lang="scss">
#app {
  font-family: Roboto, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  background: #f7f7f7;
  height: 100%;
  overflow-y: hidden;
}

.svg-tooltip {
  font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple   Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
  background: rgba(255, 255, 255, .95);
  border-radius: .1rem;
  border: 1.5px solid #333;
  color: #333;
  font-size: 16px;
  word-wrap:break-word;
  padding: .4rem .6rem;
  position: absolute;
  z-index: 300;
}

@media (min-width: 2560px) {
  .container {
      max-width: 2560px!important;
  }
}

#feature_view {
  overflow: scroll;
}
</style>
