<template>
  <div class="info-container" :style="`position: absolute; ${positioning}`">
    <svg ref="info_parent" style="width: 100%"></svg>
  </div>
</template>

<script>
// import * as vl from "vega-lite-api"
import { mapActions, mapState } from "vuex";
import * as d3 from "d3";

export default {
  name: "Feature",
  data() {
    return {
    };
  },
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    },
    render: Boolean
  },
  computed: {
    ...mapState(["data_table", "data_header"]),
  },
  methods: {
    async renderView() {
      const width = this.$refs.info_parent
        .parentNode
        .getBoundingClientRect()
        .width

      const svg = d3.select(this.$refs.info_parent);
      svg.attr("height", 400).attr("width", width);
      svg
        .append("text")
        .attr("class", 'title')
        .attr("dx", 10)
        .attr("dy", 30)
        .style("font-family", "Arial")
        .style("font-size", "15px")
        .style("font-weight", 500)
        .style("fill", "rgba(0,0,0,0.6)")
        .text('Info')
    },
    async onResize() {
      await this.renderView();
    },
  },
  mounted () {
    this.renderView()
  }
};
</script>