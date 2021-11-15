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
            md="4"
          >
            <v-card class="mx-auto" 
            >
              <v-tabs
                v-model="tab"
                background-color="transparent"
                color="primary"
              >
                <v-tab key="Features" class="text-h6">
                  Data Attributes
                </v-tab>
              </v-tabs>

              <v-tabs-items v-model="tab">
                <v-tab-item>
                  <v-card flat id="feature_view">
                    <v-img width="1000" src="./features.png"></v-img>
                  </v-card>
                </v-tab-item>
              </v-tabs-items>
            </v-card>
          </v-col>
          <v-col
            cols="12"
            md="8"
          >
            <v-card
              class="mx-auto"
            >
            <!--
              <v-list-item two-line>
                <v-list-item-content>
                  <v-list-item-title class="text-h5">
                    Rule Matrix
                  </v-list-item-title>
                  <v-list-item-subtitle>Mon, 12:30 PM, Mostly sunny</v-list-item-subtitle>
                </v-list-item-content>
              </v-list-item>
              -->
              
              <v-tabs
                background-color="transparent"
                color="primary"
              >
                <v-tab key="Features" class="text-h6">
                  Model Rules
                </v-tab>
              </v-tabs>
              <Matrix/>
            </v-card>
          </v-col>
        </v-row>
        <v-row>
          <v-col
            cols="12"
            md="12"
          >
            <v-card class="mx-auto mt-2" 
            >
              <v-tabs
                v-model="tab"
                background-color="transparent"
                color="primary"
              >
                <v-tab key="Data Table" class="text-h6">
                  Data Table
                </v-tab>
              </v-tabs>

              <v-tabs-items v-model="tab">
                <v-tab-item>
                  <v-card flat>
                    <v-data-table
                      :headers="data_header"
                      :items="data_table"
                      :items-per-page="20"
                      dense
                      class="elevation-1"
                    ></v-data-table>
                  </v-card>
                </v-tab-item>
              </v-tabs-items>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
    </v-main>
    <div class="svg-tooltip"></div>
  </v-app>
</template>

<script>
import Matrix from './components/Matrix.vue'
import { mapActions, mapGetters, mapState } from "vuex"

export default {
  name: 'App',
  components: {
    Matrix
  },
  computed: {
    ...mapGetters(['view_width']),
    ...mapState(['data_table', 'data_header'])
  },
  methods: {
    ...mapActions(['fetchRawdata', 'updateLayout', 'updateWidth', 'setReady']),
    onResize(){
      // const width = document.getElementsByTagName('body')[0].getBoundingClientRect().width
      // this.updateWidth(width)
    }
  },
  data: () => {
    return {
      drawer: null,
      tab: 'Data Table'
    }
  },
  beforeDestroy () {
    if (typeof window === 'undefined') return
    window.removeEventListener('resize', this.onResize, { passive: true })
  },
  async mounted() {
    await this.fetchRawdata()
    await this.setReady()
    await this.updateLayout()
    window.addEventListener('resize', this.onResize, { passive: true })
    this.onResize()
  }
}
</script>

<style lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  background: #f7f7f7;
}

.svg-tooltip {
  font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple   Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
  background: rgba(69,77,93,.9);
  border-radius: .1rem;
  color: #fff;
  display: block;
  font-size: 11px;
  max-width: 320px;
  padding: .2rem .4rem;
  position: absolute;
  text-overflow: ellipsis;
  white-space: pre;
  z-index: 300;
  visibility: hidden;
}

@media (min-width: 2560px) {
  .container {
      max-width: 2560px!important;
  }
}
</style>
