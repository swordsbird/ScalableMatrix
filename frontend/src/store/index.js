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
    is_ready: false,
    rulefilter: () => 1,
    crossfilter: () => 1,
    coverfilter: () => 1,
    highlighted_sample: undefined,
    instances: [],
    data_features: [],
    data_table: [],
    data_header: [],
    model_info: {},
    colorSchema: [d3.schemeTableau10[1], d3.schemeTableau10[0]].concat(d3.schemeTableau10.slice(2)),
    page_width: 1800,
    matrixview: {
      maxlen: 22,
      sort_by_cover_num: false,
      n_lines: 80,
      padding: 10,
      cell_padding: 3,
      margin: {
        top: 130,
        right: 180,
        bottom: 50,
        left: 90,
      },
      width: 1500,
      height: 800,
      coverage_width: 50,
      glyph_width: 60,
      glyph_padding: 35,
      duration: 800,
      cell: {
        header_opacity: .3,
        highlight_header_opacity: .8,
        stroke_width: .5,
        highlight_stroke_width: 1,
        stroke_color: 'gray',
        highlight_stroke_color: 'black'
      },
      feature_size_max: 2,
      focused_feature: {
        extend_width: 160,
      },
      rule_size_max: 1,
      hist_height: 4,
      order_keys: [],
      zoom_level: 0,
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
      row_height: {
        small: 2,
        medium: 6,
        large: 32,
      },
      row_padding: 2,
    },
    featureview: {
      textwidth: 125,
      maxlen: 16,
      fontweight: 400,
      padding: 15,
      column_height: 55,
      chart_height: 20,
      handle_color: '#666666',
      glyph_color: '#d62728',
      bar_color: '#888',
    },
    tableview: {
      height: 250
    },
    tooltipview: {
      width: 300,
      content: '',
      visibility: 'hidden',
      x: 100,
      y: 100,
    },
  },
  mutations: {
    setModelInfo(state, data) {
      state.model_info = data
      //console.log('model_info', data)
    },
    setDataTable(state, data) {
      const features = data.features
      const values = data.values
      const shap = data.shap
      state.data_header = features.map(d => ({ text: d, value: d }))
      state.data_table = values[0].map((_, index) => {
        let ret = {id: `customer #${index}`, _id: index }
        for (let j = 0; j < features.length; ++j) {
          ret[features[j]] = values[j][index]
        }
        return ret
      })
      state.data_shaps = shap
      // console.log(shap)
      // console.log(values)
      //document.getElementById("feature_view").appendChild(summary)
    },
    updateTooltip(state, attr) {
      for (let key in attr) {
        state.tooltipview[key] = attr[key]
      }
      if (state.tooltipview.content == '') {
        state.tooltipview.visibility = 'hidden'
      }
    },
    changeRulefilter(state, filter) {
      state.rulefilter = filter
    },
    changeCrossfilter(state, filter) {
      state.crossfilter = filter
    },
    sortLayoutRow(state, key) {
      state.matrixview.sort_by_cover_num = !state.matrixview.sort_by_cover_num
    },
    sortLayoutCol(state, key) {
      let exist = 0
      for (let i = 0; i < state.matrixview.order_keys.length; ++i) {
        if (state.matrixview.order_keys[i].key == key) {
          exist = 1
          if (state.matrixview.order_keys[i].order == 1) {
            state.matrixview.order_keys[i].order = -1
          } else if (state.matrixview.order_keys[i].order == 0) {
            state.matrixview.order_keys[i].order = 1
          } else {
            state.matrixview.order_keys.splice(i, 1)
          }
        }
      }
      if (!exist) {
        state.matrixview.order_keys.push({
          key: key, order: 1, name: state.data_features[key] ? state.data_features[key].name : key
        })
      }
    },
    changeMatrixSize (state, { width, height }) {
      state.matrixview.width = width
      state.matrixview.height = height
    },
    ready(state, status) {
      state.is_ready = status
    },
    highlight_sample(state, sample_id) {
      if (state.highlighted_sample != sample_id) {
        state.highlighted_sample = sample_id
      } else {
        state.highlighted_sample = undefined
      }
    },
    updateMatrixLayout(state) {
      if (!state.is_ready) return
      const width = state.matrixview.width - state.matrixview.margin.left - state.matrixview.margin.right
      const height = state.matrixview.height - state.matrixview.margin.top - state.matrixview.margin.bottom
      const feature_base = [1, state.matrixview.feature_size_max]
      const rule_base = [1, state.matrixview.rule_size_max]
      const colorSchema = state.colorSchema

      let features = state.data_features.filter(d => d.selected)
      /*
      const other_features = state.data_features.filter(d => !d.selected)
      features.push({
        importance: other_features.map(d => d.importance).reduce((a, b) => a + b, 0),
        indexes: other_features.map(d => d.index),
        items: other_features.map(d => d.name),
        name: 'others',
      })
      */
      const filtered_ids = new Set(
        state.data_table
          .filter(d => state.crossfilter(d))
          .map(d => d._id)
        )
      let rules = state.rules
        .filter(d => d.selected)
        .filter(d => state.rulefilter(d))
        .filter(d => {
          let count = 0
          for (let id of d.samples) {
            if (filtered_ids.has(id)) count++
          }
          return count / d.samples.length >= 0.8
        })
        
      const n_lines = state.matrixview.zoom_level > 0 ? (state.matrixview.n_lines - 15) : state.matrixview.n_lines

      let has_primary_key = false
      const preserved_keys = new Set(['coverage', 'fidelity', 'LOF'])
      if (state.matrixview.order_keys.length == 0) {
        if (state.matrixview.zoom_level == 0) {
          rules = rules.sort((a, b) => b.coverage - a.coverage)
        }
      } else {
        if (state.matrixview.zoom_level == 0) {
          has_primary_key = 1
          rules = rules.sort((a, b) => {
            for (let index = 0; index < state.matrixview.order_keys.length; ++index) {
              const key = state.matrixview.order_keys[index].key
              const order = state.matrixview.order_keys[index].order              
              if (preserved_keys.has(key)) {
                return order * (a[key] - b[key])
              } else {
                if (!a.cond_dict[key] && b.cond_dict[key]) {
                  return 1
                } else if (a.cond_dict[key] && !b.cond_dict[key]) {
                  return -1
                } else if (!a.cond_dict[key] && !b.cond_dict[key]) {
                  continue
                } else if (a.range_key[key] != b.range_key[key]) {
                  return +order * (a.range_key[key] - b.range_key[key])
                }
              }
            }
            return a.predict - b.predict
          })
        } else {
          has_primary_key = 1
          let unique_reps = [...new Set(rules.map(d => d.father))]
          let ret = []
          for (let rep of unique_reps) {
            ret = ret
            .concat(rules.filter(d => d.father == rep && d.represent))
            .concat(rules.filter(d => d.father == rep && !d.represent)
              .sort((a, b) => {
              for (let index = 0; index < state.matrixview.order_keys.length; ++index) {
                const key = state.matrixview.order_keys[index].key
                const order = state.matrixview.order_keys[index].order              
                if (preserved_keys.has(key)) {
                  return order * (a[key] - b[key])
                } else {
                  if (!a.cond_dict[key] && b.cond_dict[key]) {
                    return 1
                  } else if (a.cond_dict[key] && !b.cond_dict[key]) {
                    return -1
                  } else if (!a.cond_dict[key] && !b.cond_dict[key]) {
                    continue
                  } else {
                    return +order * (a.range_key[key] - b.range_key[key])
                  }
                }
              }
              return a.predict - b.predict
            }))
          }
          rules = ret
        }
      }
      const orderkey_items = state.matrixview.order_keys.map(d => d.key)
        .filter(d => !preserved_keys.has(d))
      const orderkey_set = new Set(orderkey_items)
      /*
      if (rules.length > n_lines) {
        // console.log('rules.length', rules.length)
        const fidelity_thres = rules.map(d => d.fidelity).sort((a, b) => b - a)[n_lines - 1]
        rules = rules.filter(d => d.fidelity >= fidelity_thres)
        // console.log('rules.length', rules.length)
      }
      */
      const samples = new Set([].concat(...rules.map(d => d.samples)))
      state.coverfilter = (d) => samples.has(d._id)
      state.primary.has_primary_key = has_primary_key
      const importance_range = d3.extent(features, d => d.importance)
      const oldFeatureScale = d3.scaleLinear()
        .domain(importance_range)
        .range(feature_base)
      let feature_sum = features.map(d => oldFeatureScale(d.importance)).reduce((a, b) => a + b)
      const extended_cols = state.matrixview.extended_cols
      const focus_extend_width = orderkey_items.length * state.matrixview.focused_feature.extend_width
      const main_width = width 
        - (state.matrixview.padding + state.matrixview.coverage_width) * extended_cols.length 
        + 2 * state.matrixview.glyph_padding
        - focus_extend_width
      const width_ratio = main_width / feature_sum
      const main_start_x = (state.matrixview.padding + state.matrixview.coverage_width) * extended_cols.filter(d => d.position < 0).length 
        + state.matrixview.glyph_padding + state.matrixview.glyph_width + 5
      const main_end_x = main_start_x + main_width + state.matrixview.padding + focus_extend_width
      const feature_range = [feature_base[0] * width_ratio, feature_base[1] * width_ratio]
      const featureScale = d3.scaleLinear()
        .domain(importance_range)
        .range(feature_range)

      const coverage_range = d3.extent(rules, d => d.coverage)
      const oldCoverageScale = d3.scaleLinear()
        .domain(coverage_range)
        .range(rule_base)
      const showHistogram = (d) => state.matrixview.zoom_level > 0 && d.represent
      //const rule_sum = rules.map(d => oldCoverageScale(d.coverage) * representScale(d.represent)).reduce((a, b) => a + b)
      //const height_ratio = rule_height / rule_sum
      //const rule_range = [rule_base[0] * height_ratio, rule_base[1] * height_ratio]
      //const instance_height = (rule_range[0] + rule_range[1]) / 2
      const instance_height = state.matrixview.row_height.medium
      // const ruleScale = d3.scaleLinear().domain(coverage_range).range(rule_range)
        
      const coverageScale = d3.scaleLinear()
        .domain([0, Math.max(...rules.map(d => d.coverage))])
        .range([0, state.matrixview.coverage_width])
        
      const fidelityScale = d3.scaleLinear()
        .domain([0, 1])
        .range([0, state.matrixview.coverage_width])
        
      const lofScale = d3.scaleLinear()
        .domain([Math.min(...rules.map(d => d.LOF)), Math.max(...rules.map(d => d.LOF))])
        .range([0.1 * state.matrixview.coverage_width, state.matrixview.coverage_width])

      const numScale = d3.scaleSqrt()
        .domain([Math.min(...rules.map(d => d.num_children)), Math.max(...rules.map(d => d.num_children))])
        .range([10, 60])
        
      const rows = []
      let y = state.matrixview.row_padding
      for (let i = 0, lastheight = 0; i < rules.length; ++i) {
        const rule = rules[i]
        const glyphheight = instance_height
        let height = state.matrixview.row_height.medium
        if (showHistogram(rule)) {
          height = state.matrixview.row_height.large
        }
        const x = 0//state.matrixview.glyph_padding//state.matrixview.padding + state.matrixview.coverage_width
        const _width = width - (state.matrixview.padding + state.matrixview.coverage_width) * 2
        const attrwidth = {
          num_children: rule.num_children,
          num: numScale(rule.num_children),
          coverage: coverageScale(rule.coverage),
          fidelity: fidelityScale(rule.fidelity),
          LOF: lofScale(rule.LOF)
        }
        const attrfill = { coverage: 'gray', fidelity: colorSchema[rule.predict], LOF: 'gray'} //'#7ec636' }
        rows.push({ x, y, lastheight, width: _width, height, glyphheight, rule, fill: colorSchema[rule.predict], attrwidth, attrfill, samples: new Set(rule.samples) })
        lastheight = height
        y += height + state.matrixview.row_padding
      }

      for (let i = 0; i < features.length; ++i) {
        const feature = features[i]
        //if (feature.name != 'others') {
        feature.count = rows.filter(d => {
          for (let cond of d.rule.conds) {
            if (cond.key == feature.index) {
              return 1
            }
          }
          return 0
        }).length
        /* } else {
          feature.count = -1
        } */
      }
      if (state.matrixview.sort_by_cover_num) {
        features = features.sort((a, b) => b.count - a.count)
      } else {
        features = features.sort((a, b) => b.importance - a.importance)
      }

      const cols = []
      const indexed_cols = []
      const feature_padding = .5
      for (let x = main_start_x, i = 0; i < features.length; ++i) {
        const feature = features[i]
        let show_axis = orderkey_set.has(feature.index)
        const width = featureScale(feature.importance)
          + (show_axis ? (state.matrixview.focused_feature.extend_width) : 0)
        //if (feature.name != 'others') {
        const scale = d3.scaleLinear().domain(feature.range).range([feature_padding, width - feature_padding])
        const item = {
          x: x, y: 0, width, height: height,
          index: feature.index,
          items: [],
          name: feature.name,
          type: feature.dtype,
          count: feature.count,
          range: feature.range,
          values: feature.values,
          scale,
          show_axis
        }
        cols.push(item)
        indexed_cols[item.index] = item
        /*} else {
          const scale = d3.scaleLinear().domain([0, 1]).range([width, width])
          const item = {
            x: x, y: 0, width, height: height,
            items: feature.items,
            index: -1,
            name: feature.name,
            count: feature.count,
            values: feature.values,
            type: 'others',
            scale,
            show_axis: false
          } 
          cols.push(item)
          for (let index of feature.indexes) {
            indexed_cols[index] = item
          }
        }*/
        x += width
      }

      for (let i = 0; i < extended_cols.length; ++i) {
        const width = state.matrixview.coverage_width
        let x = 0
        if (extended_cols[i].position < 0) {
          x = main_start_x + extended_cols[i].position * (state.matrixview.coverage_width + state.matrixview.padding)
        } else {
          x = main_end_x + extended_cols[i].position * (state.matrixview.coverage_width + state.matrixview.padding)
        }
        const item = {
          x, y: 0, width, height, index: extended_cols[i].index, name: extended_cols[i].name
        }
        cols.push(item)
        indexed_cols[item.index] = item
      }

      rows.forEach(row => {
        row.items = row.rule.conds.filter(d => indexed_cols[d.key])
        .map((d, i) => {
          const feature = indexed_cols[d.key]
          let elements = []
          if (feature.type == 'categoric') {
            const s = d.range.reduce((a, b) => a + b)
            const neg = s > d.range.length / 2
            const cond1 = row.rule.represent && state.matrixview.zoom_level > 0
            for (let j = 0; j < d.range.length; ++j) {
              const cond2 = (d.range[j] > 0) != neg
              if (cond1 && (d.range[j] > 0)|| cond2 && !cond1)
                elements.push({
                  x0: feature.scale(j),
                  x1: feature.scale(j + 1),
                  h: row.height,
                  represent: row.rule.represent,
                  fill: row.fill,
                  neg: !cond1 && neg
                })
            }
          } else {
            elements.push({
              x0: feature.scale(Math.max(feature.range[0], d.range[0])),
              x1: feature.scale(Math.min(feature.range[1], d.range[1])),
              h: row.height,
              represent: row.rule.represent,
              fill: row.fill
            })
          }
          return {
            scale: feature.scale,
            elements,
            x: feature.x,
            y: row.y,
            width: feature.width,
            height: row.height,
            fill: row.fill,
            cond: row.rule.conds[i],
            name: feature.name,
            feature: feature,
            represent: row.rule.represent,
            samples: row.rule.represent ? [...row.samples] : []
          }
        })
        row.attr = { num: row.attrwidth.num, num_children: row.attrwidth.num_children }
        row.extends = extended_cols.map((d, i) => ({
          x1: indexed_cols[d.index].x,
          x2: indexed_cols[d.index].x + row.attrwidth[d.index],
          x: indexed_cols[d.index].x,
          y: row.y,
          width: indexed_cols[d.index].width,
          height: Math.min(instance_height, row.height),
          fill: row.attrfill[d.index],
          value: row.rule[d.index],
          represent: row.rule.represent,
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
          x: state.matrixview.padding + state.matrixview.coverage_width,
          y,
        })
        y += instance_height
      }
      state.layout = { cols, rows, instances,
        height : y,
        width : width
      }
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
      console.log(paths)
      const raw_rules = paths.map((rule) => ({
        distribution: rule.distribution,
        id: rule.name,
        tree_index: rule.tree_index,
        rule_index: rule.rule_index,
        coverage: rule.coverage,
        LOF: rule.LOF,
        loyalty: (rule.coverage ** 0.5) *  rule.LOF,
        represent: rule.represent,
        father: rule.father,
        fidelity: rule.distribution[+rule.output] / rule.samples.length,
        cond_dict: rule.range,
        num_children: rule.num_children || 0,
        predict: rule.output,
        range: rule.range,
        samples: rule.samples,
        conds: Object.keys(rule.range).map(cond_key => ({
          key: cond_key,
          range: rule.range[cond_key],
        }))
      }))//.filter(d => d.fidelity > 0.6)
      for (let rule of raw_rules) {
        rule.range_key = {}
        for (let key of Object.keys(rule.range)) {
          const range = rule.range[key]
          if (range.length == 2) {
            rule.range_key[key] = (range[0] * 1e8 + range[1])
          } else {
            const neg = range.reduce((a, b) => a + b) > range.length / 2
            if (neg) {
              rule.range_key[key] = range.map(d => d ? '0' : '2').join('')
            } else {
              rule.range_key[key] = range.map(d => d ? '1' : '0').join('')
            }
          }
        }
      }
      raw_rules.forEach(d => d.selected = 1)
      state.rules = raw_rules
    },
    setZoomStatus(state, status) {
      state.matrixview.zoom_level = status
    },
    setFeatures(state, features) {
      console.log(features[0])
      const raw_features = features.map(
        (feature, feature_index) => ({
          index: feature_index,
          id: `F${feature_index}`,
          importance: feature.importance,
          range: feature.range,
          name: feature.name,
          dtype: feature.dtype,
          values: feature.values,
        }))
      raw_features.forEach((d, index) => {
        d.selected = 1//(index < raw_features.length - 3) ? 1 : 0
      })
      state.data_features = raw_features
    },
    setInstances(state, data) {
      state.instances = data
    },
  },
  getters: {
    //model_info: 'Dataset: German credit, Model: Random Forest, Original Accuracy: 82.83%, Fedility: 95.20%',
    model_info: (state) => `Dataset: ${state.model_info.dataset}, Model: ${state.model_info.model}, Original Accuracy: ${Number(state.model_info.accuracy * 100).toFixed(2)}%, Fedility: ${Number(state.model_info.info[1].fidelity_test * 100).toFixed(2)}%`,
    rule_info: (state) => `${state.matrixview.zoom_level > 0 ? 'Zoomed view' : 'Overview'}: ${state.layout.rows.length} of ${state.model_info.num_of_rules} rules`,
    topview_height: (state) => state.matrixview.height,
    filtered_data: (state) => state.data_table.filter(d => state.crossfilter(d))
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
      let resp = await axios.post(`${state.server_url}/api/explore_rules`, { idxs: data, N: ~~(state.matrixview.n_lines ) })
      commit('setRulePaths', resp.data)
      commit('setZoomStatus', true)
      commit('updateMatrixLayout')
    },
    async showRepresentRules({ commit, state }) {
      const resp = await axios.get(`${state.server_url}/api/selected_rules`, {})
      commit('setRulePaths', resp.data)
      commit('setInstances', [])
      commit('setZoomStatus', false)
      commit('updateMatrixLayout')
    },
    async orderRow({ commit }, data) {
      commit('sortLayoutRow', data)
      commit('updateMatrixLayout')
    },
    async orderColumn({ commit }, data) {
      commit('sortLayoutCol', data)
      commit('updateMatrixLayout')
    },
    async fetchRawdata({ commit, state }) {
      let resp = await axios.get(`${state.server_url}/api/model_info`, {})
      commit('setModelInfo', resp.data)
      resp = await axios.get(`${state.server_url}/api/features`, {})
      commit('setFeatures', resp.data)
      resp = await axios.get(`${state.server_url}/api/selected_rules`, {})
      commit('setRulePaths', resp.data)
      resp = await axios.post(`${state.server_url}/api/data_table`, {})
      commit('setDataTable', resp.data)
      commit('setInstances', [])
    },
    async updateMatrixLayout({ commit }) {
      commit('updateMatrixLayout')
    },
    async setReady({ commit }) {
      commit('ready', true)
    },
    async setUnready({ commit }) {
      commit('ready', true)
    },
    async updateMatrixWidth({ commit }, width) {
      commit('changeMatrixWidth', width)
    },
    async updateMatrixSize({ commit }, { width, height }) {
      commit('changeMatrixSize', { width, height })
      commit('updateMatrixLayout')
    },
    async updateRulefilter({ commit, state }, filter) {
      commit('changeRulefilter', filter)
      commit('updateMatrixLayout')
    },
    async updateCrossfilter({ commit, state }, filter) {
      commit('changeCrossfilter', filter)
      commit('updateMatrixLayout')
    },
    highlightSample({ commit }, sample_id) {
      commit('highlight_sample', sample_id)
    },
    async tooltip({ commit }, { type, data }) {
      if (type == 'show') {
        commit('updateTooltip', { visibility: 'visible' })
      } else if (type == 'hide') {
        commit('updateTooltip', { visibility: 'hidden'  })
      } else if (type == 'text') {
        commit('updateTooltip', { content: data })
      } else if (type == 'position') {
        commit('updateTooltip', { x: data.x, y: data.y })
      }
    }
  },
  modules: {
  }
})
