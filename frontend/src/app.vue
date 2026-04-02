<template>
  <div :class="['app-container', { 'dark-mode': isDark }]" :style="!isDark ? { backgroundImage: 'url(https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1920)' } : {}">
    
    <div v-if="!isLoggedIn" class="login-wrapper">
      <div class="glass-card">
        <h2 class="title">🛰️ 遥感影像智能解译平台</h2>
        <p class="subtitle">Commercial-Grade Remote Sensing Segmentation</p>
        
        <el-form :model="loginForm" style="margin-top: 30px;">
          <el-form-item>
            <el-input v-model="loginForm.username" placeholder="用户名 (admin)" :prefix-icon="User" size="large" />
          </el-form-item>
          <el-form-item>
            <el-input v-model="loginForm.password" type="password" placeholder="密码 (123456)" :prefix-icon="Lock" size="large" @keyup.enter="handleLogin" />
          </el-form-item>
          <el-button type="primary" size="large" style="width: 100%;" :loading="loggingIn" @click="handleLogin">
            安全登录
          </el-button>
        </el-form>
      </div>
    </div>

    <div v-else class="main-dashboard">
      <div class="top-nav">
        <div class="nav-left">
          <span class="logo-text">🛰️ 遥感影像地物分割指挥舱</span>
        </div>
        <div class="nav-right" style="display: flex; align-items: center; gap: 15px;">
          <el-switch v-model="isDark" inline-prompt active-text="🌙" inactive-text="☀️" />
          
          <el-select 
            v-model="currentModel" 
            placeholder="切换模型权重" 
            style="width: 170px;"
            @change="handleModelChange"
            :disabled="switchingModel"
            size="small"
          >
            <el-option label="UNet++ (轻量默认)" value="unet++" />
            <el-option label="EfficientFormer" value="efficientformer" />
            <el-option label="YOLOv11 (专家级)" value="yolov11" />
          </el-select>

          <el-tag :type="modelLoaded ? 'success' : 'danger'">{{ modelLoaded ? '核心引擎就绪' : '引擎待命' }}</el-tag>
          <el-button type="info" size="small" @click="openUserManager">
            <el-icon><User /></el-icon> 账号管理
          </el-button>
          <el-button type="danger" size="small" @click="handleLogout">退出</el-button>
        </div>
      </div>

      <div class="control-panel">
        <el-radio-group v-model="uploadMode" size="large" @change="resetUploadState">
          <el-radio-button label="single">单张影像上传</el-radio-button>
          <el-radio-button label="batch">文件夹批量上传</el-radio-button>
        </el-radio-group>

        <div v-if="uploadMode === 'single'" class="upload-btn-group" style="margin-left: 20px;">
          <input 
            ref="singleFileInput" type="file" accept="image/*,.tif,.tiff" style="display: none;" @change="handleSingleFileChange"
          />
          <el-button type="primary" size="large" @click="triggerSingleUpload">
            <el-icon><Picture /></el-icon> 载入卫星影像
          </el-button>
          <el-button type="success" size="large" :disabled="!currentSingleFile" :loading="predicting" @click="executeSinglePredict">
            <el-icon><Promotion /></el-icon> 智能解译
          </el-button>
        </div>

        <div v-if="uploadMode === 'batch'" class="upload-btn-group" style="margin-left: 20px;">
          <input 
            ref="folderInput" type="file" webkitdirectory style="display: none;" @change="handleFolderChange"
          />
          <el-button type="primary" size="large" @click="triggerFolderUpload">
            <el-icon><Folder /></el-icon> 选择影像文件夹
          </el-button>
          <el-button type="success" size="large" :disabled="batchFileList.length === 0" :loading="predicting" @click="executeBatchPredict">
            <el-icon><Promotion /></el-icon> 执行批量解译
          </el-button>
        </div>

        <div class="gsd-config" style="margin-left: auto; display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 14px; font-weight: bold;" :style="{ color: isDark ? '#ccc' : '#333' }">空间分辨率 (米/像素):</span>
          <el-input-number v-model="gsd" :precision="2" :step="0.1" :min="0.01" size="default" style="width: 130px;" />
        </div>
      </div>

      <div v-if="batchResult" class="batch-stats-card">
        <el-alert 
          :title="`批量处理完成！共 ${batchResult.total_count} 张，成功 ${batchResult.success_count} 张，失败 ${batchResult.fail_count} 张`"
          :type="batchResult.fail_count === 0 ? 'success' : 'warning'"
          show-icon style="margin-bottom: 12px;"
        />
        <p><strong>结果保存路径:</strong> {{ batchResult.save_dir }}</p>
        <p v-if="predictTime"><strong>总耗时:</strong> {{ predictTime }} 秒</p>
      </div>

      <div class="workspace">
        <div class="image-panel main-view">
          <div class="panel-header" style="justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <el-icon><Monitor /></el-icon>
              <span>可视化沙盘 {{ currentShowFilename ? `[${currentShowFilename}]` : '' }}</span>
            </div>
            
            <div v-if="resultImage" class="view-controls" style="display: flex; align-items: center; gap: 15px; width: 380px;">
              <span style="font-size: 12px; white-space: nowrap;">图层透明度:</span>
              <el-slider v-model="maskOpacity" :min="0" :max="100" style="flex: 1; margin: 0 10px;" />
              <el-switch v-model="showHeatmap" active-text="置信热力图" inactive-text="掩码图" size="small" />
            </div>
          </div>
          
          <div class="panel-content overlay-container">
            <div v-if="!originalImage" class="empty-placeholder">
              <el-empty description="等待载入光学影像" :image-size="100" />
            </div>
            <img v-if="originalImage" :src="originalImage" class="base-img" alt="底图" />
            <img v-if="resultImage" :src="showHeatmap ? heatmapImage : resultImage" class="overlay-img" :style="{ opacity: maskOpacity / 100 }" alt="叠加层" />
          </div>
        </div>
      </div>

      <div v-if="resultImage" class="analysis-panel">
        <div class="chart-card">
          <div class="panel-header">
            <el-icon><PieChart /></el-icon>
            <span>空间覆盖率统计</span>
            <span style="margin-left: auto; font-size: 14px; color: #888;">预测耗时: {{ predictTime }} 秒</span>
          </div>
          <div class="chart-content">
            <v-chart class="pie-chart" :option="pieChartOption" autoresize />
            <div class="legend-box">
              <div class="legend-title" :style="{ color: isDark ? '#ccc' : '#333' }">类别图例</div>
              <div class="legend-list">
                <div v-for="(color, name) in CLASS_COLORS" :key="name" class="legend-item">
                  <span class="legend-color" :style="{ backgroundColor: `rgb(${color.join(',')})` }"></span>
                  <span class="legend-name" :style="{ color: isDark ? '#aaa' : '#666' }">{{ name }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="report-card">
          <div class="panel-header">
            <el-icon><DataAnalysis /></el-icon>
            <span>智能测算报告</span>
          </div>
          <div class="report-content" style="overflow-y: auto;">
            <el-table :data="areaStatsData" size="small" stripe border :style="{ '--el-table-bg-color': isDark ? '#1e1e1e' : '#fff', '--el-table-tr-bg-color': isDark ? '#1e1e1e' : '#fff' }">
              <el-table-column prop="name" label="地物类别" width="100" />
              <el-table-column prop="percent" label="占比 (%)" width="80" />
              <el-table-column prop="areaSqM" label="面积 (m²)" />
              <el-table-column prop="areaSqKm" label="面积 (km²)" />
            </el-table>

            <div class="report-footer" style="margin-top: auto; padding-top: 15px;">
              <el-button type="primary" size="small" @click="downloadResult">
                <el-icon><Download /></el-icon> 导出掩码图 (PNG)
              </el-button>
              <el-button type="success" size="small" @click="downloadGeoJSON">
                <el-icon><MapLocation /></el-icon> 导出矢量数据 (GeoJSON)
              </el-button>
              <el-button type="default" size="small" @click="downloadReport">
                <el-icon><DocumentAdd /></el-icon> 导出分析报告
              </el-button>
            </div>
          </div>
        </div>
      </div>

      <el-dialog v-model="showUserDialog" title="系统账号管理" width="600px">
        <div style="margin-bottom: 15px;">
          <el-button type="primary" @click="openAddUser">新增账号</el-button>
        </div>
        <el-table :data="userList" border style="width: 100%">
          <el-table-column prop="id" label="ID" width="60" />
          <el-table-column prop="username" label="用户名" />
          <el-table-column prop="password" label="密码" />
          <el-table-column label="操作" width="150" align="center">
            <template #default="scope">
              <el-button size="small" @click="openEditUser(scope.row)">编辑</el-button>
              <el-button size="small" type="danger" @click="deleteUser(scope.row.id)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>

        <el-dialog v-model="showEditDialog" :title="isEdit ? '编辑账号' : '新增账号'" width="400px" append-to-body>
          <el-form :model="userForm" label-width="70px">
            <el-form-item label="用户名">
              <el-input v-model="userForm.username" />
            </el-form-item>
            <el-form-item label="密码">
              <el-input v-model="userForm.password" type="password" show-password />
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="showEditDialog = false">取消</el-button>
            <el-button type="primary" @click="saveUser">确认</el-button>
          </template>
        </el-dialog>
      </el-dialog>

    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'
import { User, Lock, Picture, Promotion, Folder, ArrowRightBold, Crop, PieChart, Document, Download, DocumentAdd, Monitor, DataAnalysis, MapLocation } from '@element-plus/icons-vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart as EchartsPieChart } from 'echarts/charts'
import { TitleComponent, TooltipComponent, LegendComponent } from 'echarts/components'

use([CanvasRenderer, EchartsPieChart, TitleComponent, TooltipComponent, LegendComponent])

const CLASS_COLORS = {
  '背景': [0, 0, 0],
  '农田': [0, 255, 0],
  '建筑': [255, 0, 0],
  '森林': [0, 0, 255],
  '水体': [255, 255, 0],
  '道路': [255, 0, 255]
}

// ==== 高级功能状态 ====
const isDark = ref(false)
const maskOpacity = ref(50)      // 图层透明度 (0-100)
const showHeatmap = ref(false)   // 是否显示热力图
const gsd = ref(1.0)             // 空间分辨率(米/像素)
const heatmapImage = ref('')
const currentGeoJSON = ref('')

const isLoggedIn = ref(false)
const loggingIn = ref(false)
const loginForm = reactive({ username: 'admin', password: '123456' })

const modelLoaded = ref(false)
const predicting = ref(false)
const predictTime = ref(0)
const predictStartTime = ref(0)

const currentModel = ref('unet++')
const switchingModel = ref(false)

const uploadMode = ref('single')
const singleFileInput = ref()
const folderInput = ref()

const currentSingleFile = ref(null)
const batchFileList = ref([])
const batchResult = ref(null)

const originalImage = ref('')
const resultImage = ref('')
const currentStats = ref({})
const currentShowFilename = ref('')

// 用户管理相关状态
const showUserDialog = ref(false)
const showEditDialog = ref(false)
const isEdit = ref(false)
const userList = ref([])
const userForm = reactive({ id: null, username: '', password: '' })

const api = axios.create({ baseURL: 'http://127.0.0.1:8000', timeout: 120000 })

// ==== 计算属性 ====

// 饼图配置 (适配后端新的 stats 格式: {percent: 45.2, pixels: 1234})
const pieChartOption = computed(() => {
  const data = Object.entries(currentStats.value).map(([name, valObj]) => ({
    name, 
    value: valObj.percent, 
    itemStyle: { color: `rgb(${CLASS_COLORS[name]?.join(',') || [128,128,128]})` }
  }))
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
    series: [{
      name: '地物占比', type: 'pie', radius: ['45%', '75%'], avoidLabelOverlap: false,
      itemStyle: { borderRadius: 6, borderColor: isDark.value ? '#1e1e1e' : '#fff', borderWidth: 2 },
      label: { show: true, formatter: '{b}: {c}%', color: isDark.value ? '#e0e0e0' : '#333' },
      emphasis: { label: { show: true, fontSize: 16, fontWeight: 'bold' } },
      data: data
    }]
  }
})

// 物理面积表格数据
const areaStatsData = computed(() => {
  if (Object.keys(currentStats.value).length === 0) return []
  return Object.entries(currentStats.value).map(([name, valObj]) => {
    // 面积 = 像素总数 * (单像素宽 * 单像素高)
    const areaSqM = valObj.pixels * (gsd.value * gsd.value)
    return {
      name,
      percent: valObj.percent,
      areaSqM: areaSqM.toLocaleString('en-US', { maximumFractionDigits: 1 }),
      areaSqKm: (areaSqM / 1000000).toLocaleString('en-US', { maximumFractionDigits: 4 })
    }
  }).sort((a, b) => b.percent - a.percent) // 按占比降序排列
})

const topClass = computed(() => {
  if (Object.keys(currentStats.value).length === 0) return { name: '-', value: 0 }
  const sorted = Object.entries(currentStats.value).sort((a,b) => b[1].percent - a[1].percent)
  return { name: sorted[0][0], value: sorted[0][1].percent }
})

const bottomClass = computed(() => {
  if (Object.keys(currentStats.value).length === 0) return { name: '-', value: 0 }
  const sorted = Object.entries(currentStats.value).sort((a,b) => a[1].percent - b[1].percent)
  return { name: sorted[0][0], value: sorted[0][1].percent }
})

const analysisText = computed(() => {
  if (Object.keys(currentStats.value).length === 0) return '暂无分析数据'
  const sorted = Object.entries(currentStats.value).sort((a,b) => b[1].percent - a[1].percent)
  const mainClass = sorted[0][0], mainPercent = sorted[0][1].percent
  let text = `本次预测的遥感影像中，**${mainClass}** 为主要地物类型，占比达到 ${mainPercent}%，`
  if (mainClass === '农田') text += '说明该区域以农业用地为主，具备规模化种植的地理条件。'
  else if (mainClass === '建筑') text += '说明该区域为城镇建成区，人工建筑密集，城镇化水平较高。'
  else if (mainClass === '森林') text += '说明该区域植被覆盖率高，生态环境良好，以林地生态系统为主。'
  else if (mainClass === '水体') text += '说明该区域水域面积广阔，可能为湖泊、河流或沿海区域，水资源丰富。'
  else if (mainClass === '道路') text += '说明该区域交通路网密集，交通通达性较好。'
  else text += '是该区域最主要的地表覆盖类型。'
  return text
})

onMounted(() => {
  const token = localStorage.getItem('rs_token')
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`
    const userRole = ref('user') // 记录当前登录用户的角色
    isLoggedIn.value = false // [已修复] 之前误写成了false
    checkModelStatus()
  }
})

// ==== 登录与鉴权 ====
const handleLogin = async () => {
  loggingIn.value = true
  try {
    const formData = new FormData()
    formData.append('username', loginForm.username)
    formData.append('password', loginForm.password)
    const res = await api.post('/token', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    
    if (res.data.access_token) {
      const token = res.data.access_token
      localStorage.setItem('rs_token', token)
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
      isLoggedIn.value = true
      checkModelStatus()
      ElMessage.success('登录成功')
    } else {
      ElMessage.error(res.data.error || '登录失败')
    }
  } catch (e) {
    ElMessage.error('登录失败，请检查后端服务')
  } finally {
    loggingIn.value = false
  }
}

const handleLogout = () => {
  isLoggedIn.value = false
  localStorage.removeItem('rs_token')
  resetUploadState()
  api.defaults.headers.common['Authorization'] = ''
}

const checkModelStatus = async () => {
  try {
    const res = await api.get('/health')
    modelLoaded.value = res.data.model_loaded
    if (!modelLoaded.value) ElMessage.warning('模型未加载，请在右上角切换模型')
  } catch (e) {
    console.error(e)
  }
}

const handleModelChange = async (modelValue) => {
  if (!isLoggedIn.value) return
  switchingModel.value = true
  modelLoaded.value = false 
  ElMessage.info(`正在切换至 ${modelValue} 模型，请稍候...`)
  try {
    const formData = new FormData()
    formData.append('model_name', modelValue)
    const res = await api.post('/model/switch', formData)
    if (res.data.success) {
      ElMessage.success(res.data.message)
      modelLoaded.value = true
      resetUploadState()
    } else {
      ElMessage.error(res.data.message || '模型切换失败')
      modelLoaded.value = true
    }
  } catch (e) {
    ElMessage.error('模型切换请求失败')
    modelLoaded.value = true
  } finally {
    switchingModel.value = false
  }
}

// ==== 用户管理 ====
const openUserManager = async () => { showUserDialog.value = true; fetchUsers() }
const fetchUsers = async () => { try { const res = await api.get('/users'); if (res.data.success) userList.value = res.data.data } catch (e) { ElMessage.error('获取用户列表失败') } }
const openAddUser = () => { isEdit.value = false; userForm.id = null; userForm.username = ''; userForm.password = ''; showEditDialog.value = true }
const openEditUser = (row) => { isEdit.value = true; userForm.id = row.id; userForm.username = row.username; userForm.password = row.password; showEditDialog.value = true }
const saveUser = async () => {
  if (!userForm.username || !userForm.password) return ElMessage.warning('账号密码不能为空')
  try {
    let res = isEdit.value 
      ? await api.put(`/users/${userForm.id}`, { username: userForm.username, password: userForm.password }) 
      : await api.post('/users', { username: userForm.username, password: userForm.password })
    if (res.data.success) { ElMessage.success(res.data.message); showEditDialog.value = false; fetchUsers() } 
    else ElMessage.error(res.data.message)
  } catch (e) { ElMessage.error('保存失败') }
}
const deleteUser = (id) => {
  ElMessageBox.confirm('确定要删除这个账号吗？', '警告', { confirmButtonText: '确定', cancelButtonText: '取消', type: 'warning' }).then(async () => {
    try {
      const res = await api.delete(`/users/${id}`)
      if (res.data.success) { ElMessage.success(res.data.message); fetchUsers() } 
      else ElMessage.error(res.data.message)
    } catch (e) { ElMessage.error('删除失败') }
  }).catch(() => {})
}

// ==== 预测与处理 ====
const resetUploadState = () => {
  currentSingleFile.value = null; batchFileList.value = []; batchResult.value = null;
  originalImage.value = ''; resultImage.value = ''; heatmapImage.value = ''; currentGeoJSON.value = '';
  currentStats.value = {}; currentShowFilename.value = ''; predictTime.value = 0;
  if (singleFileInput.value) singleFileInput.value.value = ''
  if (folderInput.value) folderInput.value.value = ''
}

const triggerSingleUpload = () => { resetUploadState(); singleFileInput.value.click() }

const handleSingleFileChange = (e) => {
  const file = e.target.files[0]; if (!file) return
  currentSingleFile.value = file; currentShowFilename.value = file.name
  const reader = new FileReader(); reader.onload = (ev) => { originalImage.value = ev.target.result }; reader.readAsDataURL(file)
}

const executeSinglePredict = async () => {
  if (!currentSingleFile.value) return
  predicting.value = true; predictStartTime.value = Date.now()
  try {
    const formData = new FormData(); formData.append('file', currentSingleFile.value)
    const res = await api.post('/predict/single', formData)
    if (res.data.success) {
      originalImage.value = `data:image/png;base64,${res.data.preview_base64}`
      resultImage.value = `data:image/png;base64,${res.data.result_base64}`
      heatmapImage.value = `data:image/png;base64,${res.data.heatmap_base64}`
      currentGeoJSON.value = res.data.geojson
      currentStats.value = res.data.stats
      predictTime.value = ((Date.now() - predictStartTime.value) / 1000).toFixed(2)
      ElMessage.success('智能解译完成！')
    } else ElMessage.error(res.data.message)
  } catch (e) { ElMessage.error('请求失败，请检查后端服务') } 
  finally { predicting.value = false }
}

const triggerFolderUpload = () => { resetUploadState(); folderInput.value.click() }

const handleFolderChange = (e) => {
  const files = Array.from(e.target.files)
  const validFiles = files.filter(file => {
    const lowerName = file.name.toLowerCase()
    return ['.png', '.jpg', '.jpeg', '.tif', '.tiff'].some(ext => lowerName.endsWith(ext))
  })
  if (validFiles.length === 0) {
    ElMessage.warning('文件夹中未找到有效影像文件'); if (folderInput.value) folderInput.value.value = ''; return
  }
  batchFileList.value = validFiles; ElMessage.success(`已选择 ${validFiles.length} 张有效影像`)
}

const executeBatchPredict = async () => {
  if (batchFileList.value.length === 0) return
  predicting.value = true; predictStartTime.value = Date.now()
  ElMessage.info('正在执行批量分割，请稍候...')
  try {
    const formData = new FormData()
    batchFileList.value.forEach(file => { formData.append('files', file) })
    const res = await api.post('/predict/batch', formData)
    if (res.data.success) {
      batchResult.value = res.data
      predictTime.value = ((Date.now() - predictStartTime.value) / 1000).toFixed(2)
      if (res.data.first_image) {
        originalImage.value = `data:image/png;base64,${res.data.first_image.preview_base64}`
        resultImage.value = `data:image/png;base64,${res.data.first_image.result_base64}`
        // 批量结果目前省略了 geojson 和 heatmap，如果需要可同样接入
        currentStats.value = res.data.first_image.stats
        currentShowFilename.value = res.data.first_image.filename
      }
      ElMessage.success(`批量解译完成！结果保存至 ${res.data.save_dir}`)
    } else ElMessage.error(res.data.message)
  } catch (e) { ElMessage.error('请求失败，请检查后端服务') } 
  finally { predicting.value = false }
}

// ==== 数据导出 ====
const downloadResult = () => {
  if (!resultImage.value) return
  const a = document.createElement('a'); a.href = resultImage.value
  a.download = `${currentShowFilename.value.split('.')[0]}_mask.png`; a.click()
}

const downloadGeoJSON = () => {
  if (!currentGeoJSON.value) return ElMessage.warning('暂无矢量数据')
  const blob = new Blob([currentGeoJSON.value], { type: 'application/json' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
  a.download = `${currentShowFilename.value.split('.')[0]}_vectors.geojson`; a.click()
}

const downloadReport = () => {
  if (Object.keys(currentStats.value).length === 0) return
  const reportText = `
遥感影像地物解译分析报告
========================================
影像名称：${currentShowFilename.value}
影像分辨率：${gsd.value} 米/像素
预测时间：${new Date().toLocaleString()}
预测耗时：${predictTime.value} 秒
========================================
一、地物面积测算统计
${Object.entries(currentStats.value).map(([name, valObj]) => {
  const areaSqM = (valObj.pixels * (gsd.value * gsd.value)).toFixed(2)
  return `${name}: ${valObj.percent}% (约 ${areaSqM} 平方米)`
}).join('\n')}
========================================
二、结果解读
${analysisText.value}
========================================`
  const blob = new Blob([reportText], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url
  a.download = `${currentShowFilename.value.split('.')[0]}_报告.txt`; a.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
/* ====== 全局 CSS 变量 (暗黑模式核心) ====== */
.app-container { 
  --bg-color: transparent;
  --card-bg: white;
  --text-primary: #333;
  --border-color: #e9ecef;
  
  width: 100vw; height: 100vh; 
  background-size: cover; background-position: center; 
  position: relative; overflow: hidden; 
  font-family: 'Segoe UI', sans-serif; 
}

/* 🌙 暗黑模式下的色彩定义 */
.app-container.dark-mode {
  --bg-color: #121212;
  --card-bg: #1e1e1e;
  --text-primary: #e0e0e0;
  --border-color: #333;
  background-color: var(--bg-color); /* 暗黑模式去掉背景图 */
}

/* 登录页样式 */
.login-wrapper { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; backdrop-filter: blur(5px); }
.glass-card { background: rgba(255, 255, 255, 0.85); padding: 40px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); width: 400px; text-align: center; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); }
.title { margin: 0; color: #1a73e8; font-size: 24px; }
.subtitle { margin: 8px 0 0 0; color: #5f6368; font-size: 14px; }

/* 主界面框架 */
.main-dashboard { width: 100%; height: 100%; display: flex; flex-direction: column; background: var(--bg-color); transition: background 0.3s; }
.top-nav { height: 60px; background: var(--card-bg); box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; justify-content: space-between; align-items: center; padding: 0 30px; border-bottom: 1px solid var(--border-color); }
.logo-text { font-weight: bold; font-size: 18px; color: var(--text-primary); }

.control-panel { background: var(--card-bg); margin: 16px 20px 0; padding: 16px 20px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); display: flex; align-items: center; gap: 20px; flex-wrap: wrap; border: 1px solid var(--border-color); }
.upload-btn-group { display: flex; gap: 12px; }

/* ====== 核心视图区 (叠底魔法) ====== */
.workspace { flex-shrink: 0; height: 420px; display: flex; padding: 16px 20px 0; }
.main-view { flex: 1; height: 100%; background: var(--card-bg); border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); display: flex; flex-direction: column; overflow: hidden; border: 1px solid var(--border-color); }

.panel-header { padding: 12px 16px; background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border-color); font-weight: 600; display: flex; align-items: center; color: var(--text-primary); }
.dark-mode .panel-header { background: rgba(255,255,255,0.03); }

.overlay-container { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; background: #000; overflow: hidden; }
.empty-placeholder { display: flex; align-items: center; justify-content: center; height: 100%; width: 100%; background: var(--card-bg); }

/* 底图与叠加图的绝对定位 */
.base-img, .overlay-img {
  position: absolute;
  max-width: 100%; max-height: 100%;
  object-fit: contain;
}
.overlay-img {
  pointer-events: none; /* 防止遮挡底图鼠标事件 */
  transition: opacity 0.1s ease;
}

/* ====== 分析与图表区 ====== */
.analysis-panel { flex: 1; display: flex; padding: 16px 20px 20px; gap: 16px; overflow: hidden; }
.chart-card, .report-card { background: var(--card-bg); border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); display: flex; flex-direction: column; overflow: hidden; border: 1px solid var(--border-color); }
.chart-card { flex: 1; }
.report-card { flex: 0.8; }
.chart-content { flex: 1; display: flex; padding: 16px; gap: 16px; }
.pie-chart { flex: 1; height: 100%; }

.legend-box { width: 140px; flex-shrink: 0; border-left: 1px solid var(--border-color); padding-left: 16px; }
.legend-title { font-weight: 600; margin-bottom: 12px; }
.legend-list { display: flex; flex-direction: column; gap: 10px; }
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 14px; }
.legend-color { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #eee; }

.report-content { padding: 16px; display: flex; flex-direction: column; height: 100%; }
.report-title { font-weight: 600; color: #1a73e8; margin: 0 0 12px 0; font-size: 15px; }
.report-list { margin: 0 0 16px 0; padding-left: 20px; line-height: 2; color: var(--text-primary); }
.report-text { line-height: 1.8; color: var(--text-primary); margin: 0; text-indent: 2em; opacity: 0.8; }
.report-footer { display: flex; gap: 12px; justify-content: flex-end; }
</style>