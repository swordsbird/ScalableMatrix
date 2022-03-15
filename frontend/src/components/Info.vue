<template>
  <div class="info-container ma-3" :style="`position: absolute; ${positioning}`">
    <div class="align-center tree-subtitle mb-2">
      Rule Info
    </div>
    <div class="tree-text ml-1 mt-3" v-if="summary_info.info">
      <p class="my-1">
        {{ view_info }}
      </p>
      <p class="my-1">
        The current rules view shows 
        <b>{{summary_info.number_of_rules}}</b> rules covering 
        <b>{{summary_info.info.total}}</b> samples, including 
        <span :style="`color: ${color_schema[1]}`"><b>{{summary_info.info.positives}}</b></span> positive samples and 
        <span :style="`color: ${color_schema[0]}`"><b>{{summary_info.info.total - summary_info.info.positives}}</b></span> negative samples, with a positive rate of 
        <b>{{Number(summary_info.info.positives / summary_info.info.total * 100).toFixed(2)}}%</b>.
      </p>
    </div>
    <hr class="my-2"/>
    <template v-if="zoom_level > 0">
      <div class="align-center tree-subtitle my-2">
        Suggestion
      </div>
      <template v-if="summary_info.suggestion">
        <div class="tree-text my-2" v-for="hint in summary_info.suggestion">
          If
          <b>{{hint[1][0]}} {{hint[1][1]}} {{Number(hint[1][2]).toFixed(6)}}</b>, the positive rate becomes
          <b>{{Number(hint[0] * 100).toFixed(2)}}%</b>
          <span style="color: green">&nbsp;(<b>+{{Number((hint[0] - summary_info.info.positives / summary_info.info.total) * 100).toFixed(2)}}%</b>)</span>
        </div>
      </template>
      <template v-if="!summary_info.suggestion">
        <div class="tree-text my-2" v-for="(target, index) in model_info.targets">
          <v-btn
            :loading="loading"
            class="ma-1"
            color="default"
            plain
            @click="findSuggestion(index)"
          >
            Increase
          </v-btn>
          <span
            style="font-style: italic; font-weight: 700;"
            :style="`color: ${color_schema[index]}`">
              Prob({{target}})
          </span>
        </div>
      </template>
      <hr/>
    </template>
    <div class="align-center tree-subtitle my-2">
      Legend
    </div>
  </div>
</template>

<script>
// import * as vl from "vega-lite-api"
import { mapActions, mapState, mapGetters } from "vuex";
import * as d3 from "d3";
import * as axios from 'axios'

export default {
  name: "Feature",
  data() {
    return {
      info: null,
      hint: null,
      legend: null,
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
    ...mapState(["data_table", "data_header", "summary_info", "color_schema", 'model_info' ]),
    ...mapGetters([ 'rule_info', 'view_info', 'zoom_level' ]),
  },
  methods: {
    ...mapActions(['findSuggestion']),
    async renderView() {
    },
    async onResize() {
      await this.renderView();
    }
  },
  mounted () {
    this.renderView()
  }
};
</script>
