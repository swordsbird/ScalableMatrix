<template>
  <div class="info-container ma-3" :style="`position: absolute; ${positioning}`">
    <div class="align-center tree-subtitle mb-2">
      Rule Info
    </div>
    <div class="tree-text ml-1 mt-3" v-if="summary_info.info">
      <v-card elevation="0" class="mx-auto">
        <v-card-text class="text--primary pa-1 tree-text">
          <div>{{ view_info }}</div>
          <div>
            The current rules view shows 
            <b>{{summary_info.number_of_rules}}</b> rules covering 
            <b>{{summary_info.info.total}}</b> samples, including 
            <span :style="`color: ${color_schema[1]}`"><b>{{summary_info.info.positives}}</b></span> positive samples and 
            <span :style="`color: ${color_schema[0]}`"><b>{{summary_info.info.total - summary_info.info.positives}}</b></span> negative samples, with a positive rate of 
            <b>{{Number(summary_info.info.positives / summary_info.info.total * 100).toFixed(2)}}%</b>.
          </div>
        </v-card-text>
      </v-card>
    </div>
    <template v-if="summary_info.current_rule">
    <hr class="my-2"/>
      <div class="align-center tree-subtitle my-2">
        Current Rule
      </div>
        <v-card elevation="0" class="mx-auto">
          <!--v-card-title>Top 10 Australian beaches</v-card-title>
          <v-card-subtitle class="pb-0">
            Number 10
          </v-card-subtitle-->

          <v-card-text class="text--primary pa-1 tree-text">
            <div>
              <b>if</b>
              <v-chip label v-for="(item, i) in summary_info.current_rule.items" :key="i"
                class="pa-1 ma-1"
              >
                {{item.text}}
              </v-chip>
              <b>then</b>
              <span :style="`color: ${color_schema[summary_info.current_rule.rule.predict]}`"><b>
                {{dataset.label[summary_info.current_rule.rule.predict]}}
              </b></span>
            </div>
          </v-card-text>

          <v-card-actions>
            <v-btn
              text @click="updateRuleLabel({
                name: summary_info.current_rule.id,
                label: 1
              })"
            >
              Anomaly
            </v-btn>

            <v-btn
              text @click="updateRuleLabel({
                name: summary_info.current_rule.id,
                label: 0
              })"
            >
              Non-anomaly
            </v-btn>
          </v-card-actions>
        </v-card>
      <hr/>
    </template>
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
    ...mapState(["data_table", "data_header", "summary_info", "color_schema", 'model_info', 'dataset' ]),
    ...mapGetters([ 'rule_info', 'view_info', 'zoom_level' ]),
  },
  methods: {
    ...mapActions(['findSuggestion', 'updateRuleLabel']),
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
