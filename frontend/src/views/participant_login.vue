<template>
  <div class="login-container">
    <div class="login-card">
      <div class="header">
        <h1>Participant Login</h1>
      </div>

      <form @submit.prevent="handleLogin" class="login-form">
        <div class="input-group">
          <label for="sessionName">Session Key</label>
          <input
            id="sessionName"
            v-model="sessionName"
            type="text"
            required
            placeholder="Enter the session key you received"
            :disabled="isLoading"
          />
        </div>

        <button type="submit" class="login-btn" :disabled="isLoading">
          {{ isLoading ? 'Logging in...' : 'Login' }}
        </button>
      </form>

      <div v-if="errorMessage" class="error-message">
        {{ errorMessage }}
      </div>

      <div class="info-section">
        <h3>Instructions</h3>
        <ul>
          <li>Enter the session key provided for this study</li>
          <li>Your session will be created if it does not already exist</li>
          <li>Returning with the same key continues the same session</li>
          <li>Contact the researcher if you have login issues</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

// Reactive state
const sessionName = ref('')
const isLoading = ref(false)
const errorMessage = ref('')

// Handle form submission
const handleLogin = async () => {
  if (!sessionName.value.trim()) {
    errorMessage.value = 'Please enter your session key'
    return
  }

  isLoading.value = true
  errorMessage.value = ''

  try {
    // Using sessionStorage for tab-specific authentication
    // This allows multiple participants to login simultaneously in different tabs
    const response = await fetch('/api/auth/public-session-login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_key: sessionName.value.trim()
      })
    })

    const data = await response.json()

    if (response.ok && data.success) {
      // Store authentication data in sessionStorage (tab-specific)
      sessionStorage.setItem('auth_token', data.token)
      sessionStorage.setItem('participant_id', data.participant.participant_id || data.participant.id)
      sessionStorage.setItem('session_code', data.session.session_code || data.session.session_name)
      sessionStorage.setItem('session_id', data.session.session_id)
      
      // Store experiment type if available
      if (data.session?.experiment_type) {
        sessionStorage.setItem('experiment_type', data.session.experiment_type)
      }

      // Header role badge is Map Task only — do not persist role for other experiments (e.g. wordguessing)
      if (data.session?.experiment_type === 'maptask') {
        const pr = data.participant?.role
        if (pr != null && String(pr).trim() !== '') {
          sessionStorage.setItem('participant_role', String(pr).trim())
        }
      } else {
        sessionStorage.removeItem('participant_role')
      }
      
      // Store participant interface config for participant.vue to render dynamically
      if (data.participant?.interface) {
        sessionStorage.setItem('participant_interface', JSON.stringify(data.participant.interface))
      }

      // Redirect to participant interface (now adaptive)
      router.push('/participant')
    } else {
      errorMessage.value = data.message || 'Login failed. Please check your credentials.'
    }
  } catch (error) {
    console.error('Login error:', error)
    errorMessage.value = 'Network error. Please check your connection and try again.'
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.login-container {
  box-sizing: border-box;
  /* #app has padding: 1vh top+bottom (see style.css); 100vh + that caused page scroll */
  min-height: calc(100vh - 2vh);
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 16px;
}

.login-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  padding: 28px;
  max-width: 500px;
  width: 100%;
}

.header {
  text-align: center;
  margin-bottom: 20px;
}

.header h1 {
  color: #333;
  margin-bottom: 8px;
  font-size: 28px;
}

.subtitle {
  color: #666;
  font-size: 16px;
}

.login-form {
  margin-bottom: 20px;
}

.input-group {
  margin-bottom: 16px;
}

.input-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
}

.input-group input {
  width: 100%;
  padding: 12px;
  border: 2px solid #e1e5e9;
  border-radius: 6px;
  font-size: 16px;
  transition: border-color 0.3s;
  box-sizing: border-box;
}

.input-group input:focus {
  outline: none;
  border-color: #667eea;
}

.input-group input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.login-btn {
  width: 100%;
  background: #667eea;
  color: white;
  border: none;
  padding: 14px;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.3s;
}

.login-btn:hover:not(:disabled) {
  background: #5a6fd8;
}

.login-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  background: #f8d7da;
  color: #721c24;
  padding: 12px;
  border-radius: 6px;
  margin-bottom: 20px;
  border: 1px solid #f5c6cb;
}

.info-section {
  border-top: 1px solid #e1e5e9;
  padding-top: 16px;
  margin-bottom: 0;
}

.info-section h3 {
  color: #333;
  margin-bottom: 10px;
  font-size: 18px;
}

.info-section ul {
  list-style: none;
  padding: 0;
  text-align: left;
}

.info-section li {
  margin-bottom: 6px;
  padding-left: 20px;
  position: relative;
}

.info-section li:last-child {
  margin-bottom: 0;
}

.info-section li:before {
  content: "•";
  color: #667eea;
  font-weight: bold;
  position: absolute;
  left: 0;
}

.experiment-info {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 6px;
}

.experiment-info h4 {
  color: #333;
  margin-bottom: 10px;
  font-size: 16px;
}

.experiment-info p {
  color: #666;
  line-height: 1.5;
  margin: 0;
}
</style>
