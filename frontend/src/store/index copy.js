import Vue from 'vue'
import Vuex from 'vuex'
import * as d3 from 'd3'
import * as axios from 'axios'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    server_url: 'http://166.111.81.51:5000',
    layout: null,
    primary: {
      key: null,
      order: -1,
    },
    extended_cols: [
      {
        position: 1,
        name: 'Coverage',
        index: 'coverage',
      },
      {
        position: 2,
        name: 'Anomaly Score',
        index: 'LOF',
      },
      {
        position: 0,
        name: 'Confidence',
        index: 'fidelity',
      },
    ],
    is_zoomed: false,
    is_ready: false,
    instances: [],
    features: [],
    data_table: [],
    data_header: [],
    colorSchema: [d3.schemeTableau10[1], d3.schemeTableau10[0]].concat(d3.schemeTableau10.slice(2)),
    current_width: 1800,
    setting: {
      n_lines: 100,
      rule_view_scale: 1,
      sample_view_scale: 0,
      padding: 10,
      cell_padding: 2,
      margin: {
        top: 160,
        right: 150,
        bottom: 10,
        left: 90,
      },
      width: 1500,
      max_width: 1500,
      height: 1100,
      coverage_width: 50,
      glyph_padding: 35,
      duration: 800,
      cell_stroke_width: .2,
      feature_size_max: 2,
      primary_feature_size: 2.5,
      rule_size_max: 1,
      n_feature_show_axis: 3,
    },
  },
  mutations: {
    setDataTable(state, data) {
      const features = data.map(d => d[0])
      const values = data.map(d => d[1])
      state.data_header = features.map(d => ({ text: d, value: d }))
      state.data_table = values[0].map((_, index) => {
        let ret = {}
        for (let j = 0; j < features.length; ++j) {
          ret[features[j]] = values[j][index]
        }
        return ret
      })
      //document.getElementById("feature_view").appendChild(summary)
    },
    sortLayoutCol(state, key) {
      if (state.primary.key == key) {
        if (state.primary.order == 1) {
          state.primary.order = -1
        } else if (state.primary.order == 0) {
          state.primary.order = 1
        } else {
          state.primary.key = null
        }
      } else {
        state.primary.key = key
        state.primary.order = 1
      }
    },
    changeWidth(state, width) {
      state.current_width = width
      state.setting.width = width - state.setting.padding * 2
    },
    ready(state, status) {
      state.is_ready = status
    },
    updateLayout(state) {
      if (!state.is_ready) return
      const width = state.setting.width - state.setting.margin.left - state.setting.margin.right
      const height = state.setting.height - state.setting.margin.top - state.setting.margin.bottom
      const feature_base = [1, state.setting.feature_size_max]
      const rule_base = [1, state.setting.rule_size_max]
      const colorSchema = state.colorSchema

      const features = state.features.filter(d => d.selected)
      const other_features = state.features.filter(d => !d.selected)
      features.push({
        importance: other_features.map(d => d.importance).reduce((a, b) => a + b, 0),
        indexes: other_features.map(d => d.index),
        items: other_features.map(d => d.name),
        name: 'others',
      })
      let rules = state.rules.filter(d => d.selected)
      const rule_height = height * rules.length / state.setting.n_lines

      let has_primary_key = false
      if (state.primary.key == null) {
        if (!state.is_zoomed) {
          rules = rules.sort((a, b) => b.coverage - a.coverage)
        }
      } else {
        if (state.primary.key == 'coverage') {
          rules = rules.sort((a, b) => state.primary.order * (a.coverage - b.coverage))
        } else if (state.primary.key == 'fidelity') {
          rules = rules.sort((a, b) => state.primary.order * (a.predict - b.predict))
        } else if (state.primary.key == 'LOF') {
          rules = rules.sort((a, b) => state.primary.order * (a.LOF - b.LOF))
        } else {
          has_primary_key = 1
          rules = rules.sort((a, b) => {
            if (!a.cond_dict[state.primary.key]) {
              return 1
            } else if (!b.cond_dict[state.primary.key]) {
              return -1
            } else if (a.cond_dict[state.primary.key][0] == b.cond_dict[state.primary.key][0]) {
              return +state.primary.order * (a.cond_dict[state.primary.key][1] - b.cond_dict[state.primary.key][1])
            } else {
              return +state.primary.order * (a.cond_dict[state.primary.key][0] - b.cond_dict[state.primary.key][0])
            }
          })
        }
      }
      state.primary.has_primary_key = has_primary_key
      const importance_range = d3.extent(features, d => d.importance)
      const oldFeatureScale = d3.scaleLinear()
        .domain(importance_range)
        .range(feature_base)
      let feature_sum = features.map(d => oldFeatureScale(d.importance)).reduce((a, b) => a + b)
      if (has_primary_key) {
        feature_sum += state.setting.primary_feature_size - 1
      }
      const extended_cols = state.extended_cols
      const main_width = width - (state.setting.padding + state.setting.coverage_width) * extended_cols.length - state.setting.glyph_padding
      const width_ratio = main_width / feature_sum
      const main_start_x = (state.setting.padding + state.setting.coverage_width) * extended_cols.filter(d => d.position < 0).length + state.setting.glyph_padding
      const main_end_x = main_start_x + main_width + state.setting.padding
      const feature_range = [feature_base[0] * width_ratio, feature_base[1] * width_ratio]
      const featureScale = d3.scaleLinear()
        .domain(importance_range)
        .range(feature_range)

      const coverage_range = d3.extent(rules, d => d.coverage)
      const oldCoverageScale = d3.scaleLinear()
        .domain(coverage_range)
        .range(rule_base)
      const rule_sum = rules.map(d => oldCoverageScale(d.coverage)).reduce((a, b) => a + b)
      const height_ratio = rule_height / rule_sum
      const rule_range = [rule_base[0] * height_ratio, rule_base[1] * height_ratio]
      const instance_height = (rule_range[0] + rule_range[1]) / 2
      const ruleScale = d3.scaleLinear()
        .domain(coverage_range)
        .range(rule_range)
        
      const coverageScale = d3.scaleLinear()
        .domain([0, Math.max(...rules.map(d => d.coverage))])
        .range([0, state.setting.coverage_width])
        
      const fidelityScale = d3.scaleLinear()
        .domain([0, 1])
        .range([0, state.setting.coverage_width])
        
      const lofScale = d3.scaleLinear()
        .domain([Math.min(...rules.map(d => d.LOF)), Math.max(...rules.map(d => d.LOF))])
        .range([0.1 * state.setting.coverage_width, state.setting.coverage_width])

      const numScale = d3.scaleSqrt()
        .domain([Math.min(...rules.map(d => d.num_children)), Math.max(...rules.map(d => d.num_children))])
        .range([3, 8])
        
      const rows = []
      let y = 0
      for (let i = 0; i < rules.length; ++i) {
        const rule = rules[i]
        const height = ruleScale(rule.coverage)
        const x = 0//state.setting.glyph_padding//state.setting.padding + state.setting.coverage_width
        const _width = width - (state.setting.padding + state.setting.coverage_width) * 2
        const attrwidth = {
          num: numScale(rule.num_children),
          coverage: coverageScale(rule.coverage),
          fidelity: fidelityScale(rule.fidelity),
          LOF: lofScale(rule.LOF)
        }
        const attrfill = { coverage: 'gray', fidelity: colorSchema[rule.predict], LOF: 'gray'} //'#7ec636' }
        rows.push({ x, y, width: _width, height, rule, fill: colorSchema[rule.predict], attrwidth, attrfill })
        y += height
      }

      const cols = []
      const indexed_cols = []
      for (let x = main_start_x, i = 0; i < features.length; ++i) {
        const feature = features[i]
        let show_axis = 0
        if (has_primary_key) {
          if (feature.index == state.primary.key) {
            show_axis = 1
          } else if (i < state.setting.n_feature_show_axis - 1) {
            show_axis = 1
          }
        } else if (i < state.setting.n_feature_show_axis) {
          show_axis = 1
        }
        const width = featureScale(feature.importance) +
          (has_primary_key && feature.index == state.primary.key ? width_ratio * (state.setting.primary_feature_size - 1) : 0)
        if (feature.name != 'others') {
          const scale = d3.scaleLinear().domain(feature.range).range([0, width])
          const item = {
            x: x, y: 0, width, height: height,
            index: feature.index,
            items: [],
            name: feature.name,
            scale,
            show_axis
          }
          cols.push(item)
          indexed_cols[item.index] = item
        } else {
          const scale = d3.scaleLinear().domain([0, 1]).range([width, width])
          const item = {
            x: x, y: 0, width, height: height,
            items: feature.items,
            index: -1,
            name: feature.name,
            scale,
            show_axis: false
          }
          cols.push(item)
          for (let index of feature.indexes) {
            indexed_cols[index] = item
          }
        }
        x += width
      }

      for (let i = 0; i < extended_cols.length; ++i) {
        const width = state.setting.coverage_width
        let x = 0
        if (extended_cols[i].position < 0) {
          x = main_start_x + extended_cols[i].position * (state.setting.coverage_width + state.setting.padding)
        } else {
          x = main_end_x + extended_cols[i].position * (state.setting.coverage_width + state.setting.padding)
        }
        const item = {
          x, y: 0, width, height, index: extended_cols[i].index, name: extended_cols[i].name
        }
        cols.push(item)
        indexed_cols[item.index] = item
      }

      rows.forEach(row => {
        row.items = row.rule.conds.filter(d => indexed_cols[d.key])
        .map((d, i) => ({
          x1: indexed_cols[d.key].x + indexed_cols[d.key].scale(d.range[0]),
          x2: indexed_cols[d.key].x + indexed_cols[d.key].scale(d.range[1]),
          x: indexed_cols[d.key].x,
          y: row.y,
          width: indexed_cols[d.key].width,
          height: row.height,
          fill: row.fill,
          cond: row.rule.conds[i],
        }))
        row.attr = { num: row.attrwidth.num }
        row.extends = extended_cols.map((d, i) => ({
          x1: indexed_cols[d.index].x,
          x2: indexed_cols[d.index].x + row.attrwidth[d.index],
          x: indexed_cols[d.index].x,
          y: row.y,
          width: indexed_cols[d.index].width,
          height: row.height,
          fill: row.attrfill[d.index],
          value: row.rule[d.index],
        }))
      })

      const instances = []
      for (let instance of state.instances) {
        const radius = instance_height * 0.4
        instances.push({
          id: instance.id,
          dims: instance.x.map((t, i) => ({
            x: 
              Math.min(
                indexed_cols[i].scale.range()[1] - radius,
                Math.max(indexed_cols[i].scale.range()[0] + radius,
                indexed_cols[i].scale(t))
              ),
            y: instance_height / 2,
            r: radius,
            shap: instance.shap_values[i],
            fill: colorSchema[instance.y],
          })),
          fill: colorSchema[instance.y],
          x: state.setting.padding + state.setting.coverage_width,
          y,
        })
        y += instance_height
      }
      state.layout = { cols, rows, instances }
    },
    /*
    setAllSamples(state, data) {
      state.samples = data.sort((a, b) => a.id - b.id)
    },
    setAllRules(state, data) {
      state.rules = data.sort((a, b) => a.id - b.id)
    },
    */
    // 左端点 + 右端点同时考虑在内
    // primary + secondary 双排序
    // filter by one feature (age)
    // sample background - switch - typical sample
    // more filter / legend 
    // zoom in with more space, line to encoding a sample
    // star to encoding represent rules
    // LOF => Anomaly Score
    // categorical data - explainable matrix没有，批评
    // one-hot: important. vs others.
    // representative rules need to be highlighted, 柠檬黄, stroke, extend in the front
    // interaction design
    setRulePaths(state, paths) {
      const raw_rules = paths.map((rule) => ({
        distribution: rule.distribution,
        id: rule.name,
        tree_index: rule.tree_index,
        rule_index: rule.rule_index,
        coverage: rule.coverage,
        LOF: rule.LOF,
        loyalty: (rule.coverage ** 0.5) *  rule.LOF,
        represent: rule.represent,
        fidelity: rule.distribution[+rule.output] / rule.samples.length,
        cond_dict: rule.range,
        num_children: rule.num_children || 0,
        predict: rule.output,
        samples: rule.samples,
        conds: Object.keys(rule.range).map(cond_key => ({
          key: cond_key,
          range: rule.range[cond_key],
        }))
      }))
      raw_rules.forEach(d => d.selected = 1)
      state.rules = raw_rules
    },
    setZoomStatus(state, status) {
      state.is_zoomed = status
    },
    setFeatures(state, features) {
      const raw_features = features.map(
        (feature, feature_index) => ({
          index: feature_index,
          id: `F${feature_index}`,
          importance: feature.importance,
          range: [feature.lbound, feature.rbound],
          name: feature.name,
          options: feature.options
        }))
      raw_features.forEach((d, index) => {
        d.selected = (index < raw_features.length - 5) ? 1 : 0
      })
      state.features = raw_features
    },
    setInstances(state, data) {
      state.instances = data
    },
    setRawdata(state, data) {
      const raw_rules = [].concat(...data.paths.map(
        (tree_rules, tree_index) =>
          tree_rules.map((rule, rule_index) => ({
            distribution: rule.distribution,
            id: `T${tree_index}R${rule_index}`,
            tree_index: tree_index,
            rule_index: rule_index,
            coverage: rule.samples.length / data.X.length,
            fidelity: rule.distribution[+rule.output] / rule.samples.length,
            cond_dict: rule.range,
            predict: rule.output,
            samples: rule.samples,
            conds: Object.keys(rule.range).map(cond_key => ({
              key: cond_key,
              range: rule.range[cond_key],
            }))
          }))
      ))
      const raw_features = data.features.map(
        (feature, feature_index) => ({
          index: feature_index,
          id: `F${feature_index}`,
          importance: feature.importance,
          range: [feature.lbound, feature.rbound],
          name: feature.name,
          options: feature.options
        }))
      raw_features.forEach(d => d.selected = 1)
      raw_rules.forEach(d => d.selected = 0)
      for (let i = 0; i < raw_rules.length; i += (~~(raw_rules.length / 50))) raw_rules[i].selected = 1
      state.rules = raw_rules
      state.features = raw_features
      state.instances = data.X
      // console.log('raw_rules', raw_rules)
      // console.log('raw_features', raw_features)
      state.rawdata = data
    },
  },
  getters: {
    view_width: (state) => state.setting.width,
  },
  actions: {
    /*
    async fetchAllSample({ commit, state, getters }) {
      let resp = await axios.post(`${state.server_url}/api/samples`, {})
      commit('setAllSamples', resp.data.samples)
    },
    async fetchAllRules({ commit, state, getters }) {
      let resp = await axios.post(`${state.server_url}/api/rules`, {})
      commit('setAllRules', resp.data.rules)
    },
    */
    async showExploreRules({ commit, state }, data) {
      let resp = await axios.post(`${state.server_url}/api/explore_rules`, { idxs: data, N: ~~(state.setting.n_lines * state.setting.rule_view_scale) })
      commit('setRulePaths', resp.data)
      resp = await axios.post(`${state.server_url}/api/rule_samples`, { names: resp.data.map(d => d.name), N: ~~(state.setting.n_lines * state.setting.sample_view_scale) })
      commit('setInstances', resp.data)
      commit('setZoomStatus', true)
      commit('updateLayout')
    },
    async showRepresentRules({ commit, state }) {
      const resp = await axios.get(`${state.server_url}/api/selected_rules`, {})
      commit('setRulePaths', resp.data)
      commit('setInstances', [])
      commit('setZoomStatus', false)
      commit('updateLayout')
    },
    async orderColumn({ commit }, data) {
      commit('sortLayoutCol', data)
      commit('updateLayout')
    },
    async fetchRawdata({ commit, state }) {
      let resp = await axios.get(`${state.server_url}/api/features`, {})
      commit('setFeatures', resp.data)
      resp = await axios.get(`${state.server_url}/api/selected_rules`, {})
      commit('setRulePaths', resp.data)
      resp = await axios.post(`${state.server_url}/api/data_table`, {})
      commit('setDataTable', resp.data)
      commit('setInstances', [])
    },
    async updateLayout({ commit }) {
      commit('updateLayout')
    },
    async setReady({ commit }) {
      commit('ready', true)
    },
    async setUnready({ commit }) {
      commit('ready', true)
    },
    async updateWidth({ commit, state }, width) {
      // console.log(width, state.setting.width)
      commit('changeWidth', width)
      console.log(width)
      commit('updateLayout')
    }
  },
  modules: {
  }
})
