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
    debug: true,
    is_ready: false,
    rulefilter: () => 1,
    crossfilter: () => 1,
    coverfilter: () => 1,
    highlighted_sample: undefined,
    instances: [],
    dataset: {
      name: 'german',
      format: ',.0f',
    },
    dataset_candidates: [{
      name: 'bankruptcy',
      format: ',.5f',
    }, {
      name: 'german',
      format: ',.0f',
    }],
    data_features: [],
    data_table: [],
    data_header: [],
    model_info: { loading: false },
    color_schema: [d3.schemeTableau10[1], d3.schemeTableau10[0]].concat(d3.schemeTableau10.slice(2)),
    page: { width: 1800, height: 1000 },
    covered_samples: [],
    summary_info: {
      info: null,
      suggestion: null,
    },
    matrixview: {
      maxlen: 22,
      font_size: 16,
      max_level: -1,
      sort_by_cover_num: false,
      n_lines: 80,
      padding: 10,
      cell_padding: 3,
      max_columns: 40,
      last_show_rules: [],
      margin: {
        top: 130,
        right: 180,
        bottom: 40,
        left: 75,
      },
      width: 1500,
      height: 800,
      coverage_width: 50,
      glyph_width: 60,
      glyph_padding: 35,
      bar_min_width: 2.5,
      duration: 800,
      cell: {
        feature_padding: .5,
        header_opacity: .5,
        highlight_header_opacity: .8,
        stroke_width: 1,
        highlight_stroke_width: 1,
        stroke_color: 'darkgray',
        highlight_stroke_color: 'black'
      },
      feature_size_max: 2,
      focused_feature: {
        extend_width: 160,
      },
      rule_size_max: 1,
      hist_height: 4,
      order_keys: [],
      focus_keys: [],
      zoom_level: 0,
      extended_cols: [],
      row_height: {
        small: 2,
        medium: 6.3,
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
      scrollbar_width: 10,
      handle_color: '#666666',
      glyph_color: '#d62728',
      bar_color: '#888',
      highlight_color: d3.color('#df5152').darker(-0.3),
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
      state.model_info.loaded = true
      const extended_cols = [['Confidence', 'fidelity'], ['Coverage', 'coverage'], ['Anomaly Score', 'LOF']]
      if (state.model_info.weighted) {
        extended_cols.splice(2, 0, ['Weight', 'weight'])
      }
      state.matrixview.extended_cols = extended_cols.map((d, position) => ({
        position, name: d[0], index: d[1],
      }))
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
    setSuggestion(state, suggestion) {
      state.summary_info.suggestion = suggestion
    },
    focusLayoutCol(state, key) {
      if (!state.data_features[key]) return
      let exist = 0
      for (let i = 0; i < state.matrixview.focus_keys.length; ++i) {
        if (state.matrixview.focus_keys[i].key == key) {
          exist = 1
          state.matrixview.focus_keys.splice(i, 1)
        }
      }
      if (!exist) {
        state.matrixview.focus_keys.push({
          key: key, name: state.data_features[key].name
        })
      }
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
    changePageSize (state, { width, height }) {
      state.page.width = width
      state.page.height = height
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
      const color_schema = state.color_schema

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
        /*
        .filter(d => {
          let count = 0
          for (let id of d.samples) {
            if (filtered_ids.has(id)) count++
          }
          return count / d.samples.length >= 0.8
        })
        */
        
      // const n_lines = state.matrixview.zoom_level > 0 ? (state.matrixview.n_lines - 15) : state.matrixview.n_lines

      let has_primary_key = false
      const max_level = Math.max(...rules.map(d => d.level))
      const min_level = Math.min(...rules.map(d => d.level))
      state.matrixview.zoom_level = max_level
      // console.log('level', max_level, min_level)
      rules.forEach(d => {
        if (d.level == min_level) {
          d.represent = true
          d.show_hist = (state.matrixview.zoom_level > 0)
        } else {
          d.represent = false
          d.show_hist = false
        }
      })
      //console.log('rules', rules)
      //console.log()
      const preserved_keys = new Set(state.matrixview.extended_cols.map(d => d.index))
      if (state.matrixview.order_keys.length == 0) {
        if (state.matrixview.zoom_level == 0) {
          if (state.model_info.weighted) {
            rules = rules.sort((a, b) => b.weight - a.weight)
          } else {
            rules = rules.sort((a, b) => b.coverage - a.coverage)
          }
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
            if (rep == -1) continue
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
                    } else if (a.range_key[key] != b.range_key[key]) {
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
      const focuskey_set = new Set(state.matrixview.focus_keys.map(d => d.key))
      /*
      if (rules.length > n_lines) {
        // console.log('rules.length', rules.length)
        const fidelity_thres = rules.map(d => d.fidelity).sort((a, b) => b - a)[n_lines - 1]
        rules = rules.filter(d => d.fidelity >= fidelity_thres)
        // console.log('rules.length', rules.length)
      }
      */
      for (let i = 0; i < features.length; ++i) {
        const feature = features[i]
        //if (feature.name != 'others') {
        feature.count = rules.filter(d => {
          for (let cond of d.conds) {
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
      // console.log(features)
      // console.log(rules)
      let covered_samples = new Set()
      for (let rule of rules) {
        for (let id of rule.samples) {
          if (!covered_samples.has(id)) {
            covered_samples.add(id)
          }
        }
      }
      state.covered_samples = [...covered_samples]

      features.forEach(d => { d.show = 1 })
      features.forEach(d => {
        if (d.count == 0) d.show = 0
      })
      const n_show_features = features.map(d => d.show).reduce((a, b) => a + b)
      if (state.matrixview.max_columns < n_show_features) {
        let delta = n_show_features - state.matrixview.max_columns
        for (let i = features.length - 1; i >= 0 && delta > 0; --i) {
          if (features[i].show) {
            features[i].show = 0
            delta -= 1
          }
        }
      }

      const samples = new Set([].concat(...rules.map(d => d.samples)))
      state.coverfilter = (d) => samples.has(d._id)
      state.primary.has_primary_key = has_primary_key
      const importance_range = d3.extent(features, d => d.importance)
      const oldFeatureScale = d3.scaleLinear()
        .domain(importance_range)
        .range(feature_base)
      let feature_sum = features.filter(d => d.show).map(d => oldFeatureScale(d.importance)).reduce((a, b) => a + b)
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
      //const rule_sum = rules.map(d => oldCoverageScale(d.coverage) * representScale(d.represent)).reduce((a, b) => a + b)
      //const height_ratio = rule_height / rule_sum
      //const rule_range = [rule_base[0] * height_ratio, rule_base[1] * height_ratio]
      //const instance_height = (rule_range[0] + rule_range[1]) / 2
      const instance_height = height / state.matrixview.n_lines - state.matrixview.row_padding//state.matrixview.row_height.medium
      // const ruleScale = d3.scaleLinear().domain(coverage_range).range(rule_range)
        
      const coverageScale = d3.scaleLinear()
        .domain([0, Math.max(...rules.map(d => d.coverage))])
        .range([0, state.matrixview.coverage_width])
        
      let fidelityScale = d3.scaleLinear()
        .domain([0, 1])
        .range([0, state.matrixview.coverage_width])
        //.range([state.matrixview.coverage_width * 0.75, state.matrixview.coverage_width])
      if (state.model_info.weighted) {
        // fidelityScale.range([state.matrixview.coverage_width * 0.75, state.matrixview.coverage_width])
      }

      const lofScale = d3.scalePow(2)
        .domain([Math.min(...rules.map(d => d.LOF)), Math.max(...rules.map(d => d.LOF))])
        .range([0.1 * state.matrixview.coverage_width, state.matrixview.coverage_width])

      const weightScale = d3.scaleSqrt()
        .domain([Math.min(...rules.map(d => d.weight)), Math.max(...rules.map(d => d.weight))])
        .range([0.1 * state.matrixview.coverage_width, state.matrixview.coverage_width])

      const numScale = d3.scaleSqrt()
        .domain([Math.min(...rules.map(d => d.num_children)), Math.max(...rules.map(d => d.num_children))])
        .range([10, 60])
        
      const rows = []
      let y = state.matrixview.row_padding
      if (rules.length * state.matrixview.row_height.large < height) {
        rules.forEach(d => d.show_hist = 1)
      }
      for (let i = 0, lastheight = 0; i < rules.length; ++i) {
        const rule = rules[i]
        const glyphheight = instance_height
        let height = instance_height//state.matrixview.row_height.medium
        if (rule.show_hist) {
          height = state.matrixview.row_height.large
        }
        const x = 0//state.matrixview.glyph_padding//state.matrixview.padding + state.matrixview.coverage_width
        const _width = width - (state.matrixview.padding + state.matrixview.coverage_width) * 2
        const attrwidth = {
          num_children: rule.num_children,
          num: numScale(rule.num_children),
          coverage: coverageScale(rule.coverage),
          fidelity: fidelityScale(rule.fidelity),
          LOF: lofScale(rule.LOF),
          weight: weightScale(rule.weight),
        }
        const attrfill = {
          coverage: 'gray',
          fidelity: color_schema[rule.predict],
          LOF: 'gray',
          weight: 'gray',
        } //'#7ec636' }
        rows.push({
          x, y, lastheight,
          width: _width, height, glyphheight,
          rule, fill: color_schema[rule.predict],
          attrwidth, attrfill,
          id: rule.id,
          samples: new Set(rule.samples)
        })
        lastheight = height
        y += height + state.matrixview.row_padding
      }

      if (state.matrixview.sort_by_cover_num || state.matrixview.zoom_level == 0) {
        features = features.sort((a, b) => b.importance - a.importance)
        features.forEach((d, i) => d.last_rank = i)
        features = features.sort((a, b) => b.count - a.count)
        features.forEach((d, i) => d.current_rank = i)
        const rank_delta = features.map(d => d.last_rank - d.current_rank).sort((a, b) => b - a)
        const rank_delta_top5 = Math.max(4, rank_delta[4])
        // console.log('rank_delta_top5', rank_delta_top5)
        features.forEach(d => {
          if (state.matrixview.zoom_level != 0 && d.current_rank + rank_delta_top5 <= d.last_rank) {
            d.hint_change = 1
          } else {
            d.hint_change = 0
          }
        })
      } else {
        features = features.sort((a, b) => b.importance - a.importance)
        features.forEach(d => {
          d.hint_change = 0
        })
      }

      const cols = []
      const indexed_cols = []
      const feature_padding = state.matrixview.cell.feature_padding
      console.log('features', features)
      for (let x = main_start_x, i = 0; i < features.length; ++i) {
        const feature = features[i]
        let show_axis = orderkey_set.has(feature.index)
        const width = featureScale(feature.importance)
          + (show_axis ? (state.matrixview.focused_feature.extend_width) : 0)
        //if (feature.name != 'others') {
        let range = feature.dtype == "category" ? 
          [0, feature.values.length] : 
          [0, feature.range[1]]
        let scale = d3.scaleLinear()
          .domain(range)
          .range([0, width - feature_padding * 2])
        if (feature.dtype != "category" && (
          feature.q[2] < (feature.range[1] - feature.range[0]) * 0.1 + feature.range[0] ||
          feature.q[2] > (feature.range[1] - feature.range[0]) * 0.9 + feature.range[0])) {
            //range = [feature.q[0], feature.q[2], feature.q[4]]
            scale = d3.scaleLinear()
              .domain([feature.q[0], feature.q[2], feature.q[4]])
              .range([0, width / 2 - feature_padding, width - feature_padding * 2])
        }
        const item = {
          x: x, y: 0, 
          width,
          height: height,
          index: feature.index,
          items: [],
          name: feature.name,
          type: feature.dtype,
          count: feature.count,
          hint_change: feature.hint_change,
          delta: feature.hint_change ? (feature.last_rank - feature.current_rank) : 0,
          range,
          values: feature.values,
          scale,
          show_axis,
          show: feature.show,
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
        if (feature.show) {
          x += width
        }
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
          x, y: 0, width, height, index: extended_cols[i].index, name: extended_cols[i].name, show: 1,
        }
        cols.push(item)
        indexed_cols[item.index] = item
      }

      rows.forEach(row => {
        row.items = row.rule.conds.filter(d => indexed_cols[d.key])
        .map((d, i) => {
          const feature = indexed_cols[d.key]
          let elements = []
          if (feature.type == 'category') {
            const s = d.range.reduce((a, b) => a + b)
            const neg = 0//s > d.range.length / 2
            const cond1 = row.rule.represent && state.matrixview.zoom_level > 0
            for (let j = 0; j < d.range.length; ++j) {
              const cond2 = (d.range[j] > 0) != neg
              if (cond1 && (d.range[j] > 0)|| cond2 && !cond1)
                elements.push({
                  x0: feature.scale(j) + feature_padding,
                  x1: feature.scale(j + 1) + feature_padding,
                  show: feature.show,
                  h: row.height,
                  show_hist: row.rule.show_hist,
                  fill: row.fill,
                  neg: !cond1 && neg
                })
            }
          } else {
            let x0 = feature.scale(Math.max(feature.range[0], d.range[0])) + feature_padding
            let x1 = feature.scale(Math.min(feature.range[1], d.range[1])) + feature_padding
            x1 = Math.max(x0 + state.matrixview.bar_min_width, x1)
            //todo
            elements.push({
              x0: x0,
              x1: x1,
              h: row.height,
              show: feature.show,
              show_hist: row.rule.show_hist,
              fill: row.fill
            })
          }
          return {
            scale: feature.scale,
            elements,
            x: feature.x,
            y: row.y,
            show: feature.show,
            width: feature.width,
            height: row.height,
            fill: row.fill,
            cond: row.rule.conds[i],
            name: feature.name,
            id: row.id,
            feature: feature,
            represent: row.rule.represent,
            show_hist: row.rule.show_hist,
            samples: row.rule.show_hist ? [...row.samples] : []
          }
        })
        row.attr = { num: row.attrwidth.num, num_children: row.attrwidth.num_children }
        row.extends = extended_cols.map((d, i) => ({
          x1: indexed_cols[d.index].x,
          x2: indexed_cols[d.index].x + row.attrwidth[d.index],
          x: indexed_cols[d.index].x,
          y: row.y,
          show: indexed_cols[d.index].show,
          width: indexed_cols[d.index].width,
          height: Math.min(instance_height, row.height),
          fill: row.attrfill[d.index],
          value: row.rule[d.index],
          represent: row.rule.represent,
        }))
      })

      // console.log('current zoom level', state.matrixview.zoom_level)
      state.layout = {
        cols, rows,
        height : y,
        width : width,
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
    setRulePaths(state, { paths, samples, info }) {
      state.summary_info.info = info
      Object.assign(state.summary_info, { suggestion: null })
      const raw_rules = paths.map((rule) => ({
        distribution: rule.distribution,
        id: rule.name,
        tree_index: rule.tree_index,
        rule_index: rule.rule_index,
        coverage: rule.coverage,
        LOF: rule.LOF,
        weight: rule.weight,
        level: rule.level,
        loyalty: (rule.coverage ** 0.5) * rule.LOF,
        father: rule.father,
        fidelity: rule.distribution[+rule.output] / rule.distribution.reduce((a, b) => a + b),
        cond_dict: rule.range,
        num_children: rule.num_children || 0,
        predict: rule.output,
        range: rule.range,
        q: rule.q,
        samples: rule.samples,
        conds: Object.keys(rule.range).map(cond_key => ({
          key: cond_key,
          range: rule.range[cond_key],
        }))
      }))//.filter(d => d.fidelity > 0.6)
      if (state.matrixview.max_level == -1) {
        state.matrixview.max_level = Math.max(...raw_rules.map(d => d.level))
      }
      for (let rule of raw_rules) {
        rule.level = state.matrixview.max_level - rule.level
        rule.range_key = {}
        for (let key of Object.keys(rule.range)) {
          const range = rule.range[key]
          if (range.length == 2) {
            rule.range_key[key] = (range[0] * 1e8 + range[1])
          } else {
            const neg = 0 // range.reduce((a, b) => a + b) > range.length / 2
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
      state.summary_info.number_of_rules = raw_rules.length
      console.log(state.summary_info)
    },
    setZoomStatus(state, status) {
      state.matrixview.zoom_level = status
    },
    setFeatures(state, features) {
      //console.log(features[0])
      const raw_features = features.map(
        (feature, feature_index) => ({
          index: feature_index,
          id: `F${feature_index}`,
          importance: feature.importance,
          range: feature.range,
          q: feature.q,
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
    model_target: (state) => state.model_info.target,
    zoom_level: (state) => state.matrixview.zoom_level,
    //model_info: 'Dataset.name: German credit, Model: Random Forest, Original Accuracy: 82.83%, Fedility: 95.20%',
    model_info: (state) => `Dataset: ${state.model_info.dataset}, Model: ${state.model_info.model}, Original Accuracy: ${Number(state.model_info.accuracy * 100).toFixed(2)}%, Fedility: ${Number(state.model_info.info[1].fidelity_test * 100).toFixed(2)}%`,
    rule_info: (state) => `${state.layout.rows.length} out of ${state.model_info.num_of_rules} rules`,
    view_info: (state) => `${state.matrixview.zoom_level > 0 ? ('Zoom level ' + state.matrixview.zoom_level) : 'Overview'}`,
    topview_height: (state) => state.matrixview.height,
    filtered_data: (state) => {
      let covered_samples = new Set(state.covered_samples)
      return state.data_table.filter(d => state.crossfilter(d) && covered_samples.has(d._id))
    },
    rule_related_data: (state) => {
      let covered_samples = new Set(state.covered_samples)
      return state.data_table.filter(d => covered_samples.has(d._id))
    },
  },
  actions: {
    async showRules({ commit, state }, data) {
      let resp
      if (data == null && state.matrixview.last_show_rules.length > 1) {
        data = state.matrixview.last_show_rules.splice(-2, 1)[0]
      } else if (data != null) {
        state.matrixview.last_show_rules.push(data)
      }
      if (data == null) {
        state.matrixview.last_show_rules = []
        resp = await axios.get(`${state.server_url}/api/selected_rules`, { params : { dataname: state.dataset.name } })
      } else {
        resp = await axios.post(`${state.server_url}/api/explore_rules`, { dataname: state.dataset.name, idxs: data, N: ~~(state.matrixview.n_lines ) })
      }
      commit('setRulePaths', resp.data)
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
      let resp = await axios.get(`${state.server_url}/api/model_info`, { params : { dataname: state.dataset.name } })
      commit('setModelInfo', resp.data)
      resp = await axios.get(`${state.server_url}/api/features`, { params : { dataname: state.dataset.name } })
      commit('setFeatures', resp.data)
      resp = await axios.get(`${state.server_url}/api/selected_rules`, { params : { dataname: state.dataset.name } })
      commit('setRulePaths', resp.data)
      resp = await axios.post(`${state.server_url}/api/data_table`, { dataname: state.dataset.name })
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
    async updatePageSize({ commit }, { width, height }) {
      commit('changePageSize', { width, height })
    },
    async updateRulefilter({ commit, state }, filter) {
      commit('changeRulefilter', filter)
      // commit('updateMatrixLayout')
    },
    async updateCrossfilter({ commit, state }, filter) {
      commit('changeCrossfilter', filter)
      // commit('updateMatrixLayout')
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
    },
    async findSuggestion({ state, commit }, target) {
      let resp = await axios.post(`${state.server_url}/api/suggestions`, {
        dataname: state.dataset.name,
        ids: state.rules.map(d => d.id),
        target: target,
      })
      commit('setSuggestion', resp.data)
    }
  },
  modules: {
  }
})
