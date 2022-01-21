<template>
  <div class="feature-container" :style="`position: absolute; ${positioning}; overflow-y: scroll`" v-resize="onResize">
    <svg ref="feature_parent" style="width: 100%"></svg>
  </div>
</template>

<script>
// import * as vl from "vega-lite-api"
import { mapActions, mapGetters, mapState } from "vuex";
import * as d3 from "d3";
import BrushableBarchart from "../libs/brushablechart";

export default {
  name: "Feature",
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    },
    render: Boolean
  },
  computed: {
    ...mapState(["covered_samples", "data_table", "data_header", "featureview", "rules"]),
    ...mapGetters(['rule_related_data'])
  },
  data() {
    return {
      model_feature_view: null,
      data_feature_view: null,
      model_features: null,
      data_features: null,
    }
  },
  watch: {
    covered_samples(val) {
      this.update()
    }
  },
  methods: {
    ...mapActions(['tooltip', 'updateCrossfilter', 'updateRulefilter']),
    renderView() {
      const width = this.$refs.feature_parent
        .parentNode
        .getBoundingClientRect()
        .width
      const self = this
      const featureview = this.featureview
      const data_features = Object.keys(this.data_table[0])
        .filter(d => d != 'id' && d != '_id')
        .map(d => ({ name: d, key: d, filter: () => 1 }))
      const model_features = [
        { name: 'Confidence', key: 'fidelity', filter: () => 1 },
        { name: 'Coverage', key: 'coverage', filter: () => 1 },
        { name: 'Anomaly Score', key: 'LOF', filter: () => 1 },
      ]
      let model_feature_height = (model_features.length + 1) * featureview.column_height
      let data_feature_height = (data_features.length + 1) * featureview.column_height
      
      const svg = d3.select(this.$refs.feature_parent)
      svg.selectAll("*").remove()
      const height = model_feature_height + data_feature_height
      svg.attr("height", height)
        .attr("width", width)

      let model_feature_view = svg.append('g')
        .attr('class', 'model-feature')
        .attr('transform', `translate(${0},${0})`)

      let data_feature_view = svg.append('g')
        .attr('class', 'data-feature')
        .attr('transform', `translate(${0},${model_feature_height})`)

      model_feature_view
        .append("text")
        .attr("class", 'title')
        .attr("dx", 10)
        .attr("dy", 30)
        .style("font-family", "Arial")
        .style("font-size", "15px")
        .style("font-weight", 500)
        .style("fill", "rgba(0,0,0,0.6)")
        .text('Model Features')
      
      data_feature_view
        .append("text")
        .attr("class", 'title')
        .attr("dx", 10)
        .attr("dy", 30)
        .style("font-family", "Arial")
        .style("font-size", "15px")
        .style("font-weight", 500)
        .style("fill", "rgba(0,0,0,0.6)")
        .text('Data Features')

      model_feature_view = model_feature_view.append('g')
        .attr('class', 'content')

      data_feature_view = data_feature_view.append('g')
        .attr('class', 'content')

      this.model_feature_view = model_feature_view
      this.data_feature_view = data_feature_view
      this.model_features = model_features
      this.data_features = data_features
      this.update()
    },
    update() {
      console.log('update with', this.covered_samples, this.rule_related_data)
      const self = this
      const featureview = this.featureview
      const width = this.$refs.feature_parent
        .parentNode
        .getBoundingClientRect()
        .width

      drawCharts(
        this.model_feature_view,
        this.rules,
        this.model_features,
        (filter) => this.updateRulefilter(filter), false
      )
      drawCharts(
        this.data_feature_view,
        this.rule_related_data,
        this.data_features, 
        (filter) => this.updateCrossfilter(filter)
      )

      function drawCharts(selection, data, features, update, brushable = true) {
        selection.selectAll('*').remove()

        selection
          .selectAll(".chart")
          .data(features)
          .enter()
          .append("g")
          .attr("class", "chart")
        
        let chart_row = selection
          .selectAll(".chart")
          .data(features)
          .attr("transform", (d, i) => `translate(${featureview.padding}, ${(i + 0.5) * featureview.column_height})`);

        chart_row
          .append("text")
          .attr("class", "name")
          .attr("dy", featureview.column_height - 10)
          .style("font-family", "Arial")
          .style("font-size", "16px")
          .style("font-weight", featureview.fontweight)
          .text((d) => (d.name.length < featureview.maxlen ? d.name : d.name.slice(0, featureview.maxlen) + "..."))
          .on("mouseover", function(ev, d){
            if (d.name.length < featureview.maxlen) return
            self.tooltip({ type: "show" })
          })
          .on("mouseout", function(ev, d){
            if (d.name.length < featureview.maxlen) return
            self.tooltip({ type: "hide" })
          })
          .on("mousemove", function(ev, d){
            if (d.name.length < featureview.maxlen) return
            self.tooltip({
              type: "text",
              data: `${d.name}`
            })
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
          })
          
        const chart_body = chart_row
          .append("g")
          .attr("class", "chart")
          .attr("transform", `translate(${featureview.textwidth}, 0)`);
        
        chart_body.each(function(d) {
          const chart = BrushableBarchart()
            .data(data)
            .x(d.key)
            .width(width - featureview.padding * 4 - featureview.textwidth)
            .height(featureview.chart_height)
            .brushable(brushable)
            .colors({
              handle: featureview.handle_color,
              glyph: featureview.glyph_color,
              bar: featureview.bar_color 
            })
            .mousemove(function(ev, d) {
              self.tooltip({ type: "show" })
              self.tooltip({ type: "text", data: `${d.name}, ${d.count}`})
              self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
            })
            .mouseout(function(ev, d){
              self.tooltip({ type: "hide" })
            })

          chart.brushend(function(){
            d.filter = chart.filter()
            const filter = (d) => {
              for (let feature of features) {
                if (!feature.filter(d)) return 0
              }
              return 1
            }
            update(filter)
          })

          d3.select(this).call(chart)
        });
      }
    },
    onResize() {
      this.renderView()
    },
  },
  mounted () {
    this.renderView()
  }
};
</script>