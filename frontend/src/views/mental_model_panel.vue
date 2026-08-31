<script setup>
import { computed, reactive, ref, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: Object,
    default: () => ({}),
  },
})

const emit = defineEmits(['update:modelValue', 'field-edit'])

const sections = [
  {
    id: 'task',
    title: 'TASK MODEL',
    accent: 'task',
    fields: [
      { key: 'taskGoal', label: 'W1 · Task goal', placeholder: 'Describe the shared task goal' },
      { key: 'taskSpecification', label: 'W2 · Task specification', placeholder: 'Describe the task requirements' },
      { key: 'procedure', label: 'W3 · Procedure', placeholder: 'Describe the agreed procedure' },
      { key: 'constraint', label: 'W4 · Constraint', placeholder: 'Add relevant constraints' },
      { key: 'decisionPriority', label: 'W5 · Decision priority', placeholder: 'State the current priority' },
      { key: 'taskState', label: 'W6 · Task state', placeholder: 'Describe the current task state' },
    ],
  },
  {
    id: 'equipment',
    title: 'EQUIPMENT MODEL',
    accent: 'equipment',
    fields: [
      { key: 'operationalCapability', label: 'W7 · Operational capability', placeholder: 'What can the system do?' },
      { key: 'informationAccess', label: 'W8 · Information access', placeholder: 'What information is available?' },
    ],
  },
  {
    id: 'interaction',
    title: 'TEAM INTERACTION',
    accent: 'interaction',
    fields: [
      { key: 'roleResponsibility', label: 'W9 · Role responsibility', placeholder: 'Who owns which responsibility?' },
      { key: 'coordinationProtocol', label: 'W10 · Coordination protocol', placeholder: 'How should the team coordinate?' },
      { key: 'communicationProtocol', label: 'W11 · Communication protocol', placeholder: 'What convention should we use to communicate?' },
    ],
  },
  {
    id: 'member',
    title: 'TEAM MEMBER',
    accent: 'member',
    fields: [
      { key: 'partnerKnowledge', label: 'W12 · Partner knowledge', placeholder: 'What does your partner know?' },
      { key: 'partnerNextAction', label: 'W13 · Partner next action', placeholder: 'What will your partner do next?' },
    ],
  },
]

const createModel = (value = {}) =>
  sections.flatMap((section) => section.fields).reduce((model, field) => {
    const raw = value[field.key]
    model[field.key] = raw && typeof raw === 'object' ? (raw.value ?? '') : (raw ?? '')
    return model
  }, {})

const fieldMeta = computed(() => {
  const result = {}
  for (const section of sections) {
    for (const field of section.fields) {
      const raw = props.modelValue?.[field.key]
      result[field.key] = raw && typeof raw === 'object'
        ? { status: raw.status || 'inferred', confidence: raw.confidence || 'low' }
        : { status: 'inferred', confidence: 'low' }
    }
  }
  return result
})

const model = reactive(createModel(props.modelValue))
const openSections = ref([])

watch(
  () => props.modelValue,
  (value) => Object.assign(model, createModel(value)),
  { deep: true },
)

const toggleSection = (sectionId) => {
  openSections.value = openSections.value.includes(sectionId)
    ? openSections.value.filter((id) => id !== sectionId)
    : [...openSections.value, sectionId]
}

const isOpen = (sectionId) => openSections.value.includes(sectionId)

const emitModel = (key) => {
  emit('update:modelValue', { ...model })
  emit('field-edit', { key, value: model[key] })
}
</script>

<template>
  <section class="mental-model-panel" aria-label="Shared Mental Model">
    <header class="panel-title">
      <span>Shared Mental Model</span>
      <span class="panel-hint">Expand a category to edit</span>
    </header>

    <div class="model-sections">
      <article
        v-for="section in sections"
        :key="section.id"
        class="model-section"
        :class="[section.accent, { open: isOpen(section.id) }]"
      >
        <button
          type="button"
          class="section-toggle"
          :aria-expanded="isOpen(section.id)"
          :aria-controls="`${section.id}-fields`"
          @click="toggleSection(section.id)"
        >
          <span>{{ section.title }}</span>
          <span class="toggle-icon" aria-hidden="true">{{ isOpen(section.id) ? '⌃' : '⌄' }}</span>
        </button>

        <div v-if="isOpen(section.id)" :id="`${section.id}-fields`" class="section-fields">
          <label v-for="field in section.fields" :key="field.key" class="model-field">
            <span class="field-label">
              {{ field.label }}
              <small>{{ fieldMeta[field.key].status }} · {{ fieldMeta[field.key].confidence }}</small>
            </span>
            <textarea
              v-model="model[field.key]"
              :placeholder="field.placeholder"
              rows="2"
              @input="emitModel(field.key)"
            />
          </label>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.mental-model-panel {
  flex: 0 0 auto;
  overflow: hidden;
  background: #ffffff;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 12px;
  color: #495057;
  background: #e9ecef;
  border-bottom: 1px solid #dee2e6;
  font-size: 14px;
  font-weight: 600;
}

.panel-hint {
  color: #6b7280;
  font-size: 10px;
  font-weight: 400;
  white-space: nowrap;
}

.model-sections {
  display: grid;
  gap: 6px;
  padding: 8px;
}

.model-section {
  overflow: hidden;
  border: 1px solid #dee2e6;
  border-left: 4px solid var(--section-accent);
  border-radius: 5px;
  background: #ffffff;
}

.model-section.task { --section-accent: #667eea; }
.model-section.equipment { --section-accent: #d2692f; }
.model-section.interaction { --section-accent: #32847e; }
.model-section.member { --section-accent: #d89b27; }

.model-section.open {
  border-color: var(--section-accent);
}

.section-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  min-height: 34px;
  padding: 7px 8px;
  color: var(--section-accent);
  background: #ffffff;
  border: 0;
  border-radius: 0;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-align: left;
}

.model-section.open .section-toggle {
  background: color-mix(in srgb, var(--section-accent) 8%, #ffffff);
}

.section-toggle:hover {
  background: color-mix(in srgb, var(--section-accent) 12%, #ffffff);
}

.section-toggle:focus-visible,
.model-field textarea:focus-visible {
  outline: 2px solid #2563eb;
  outline-offset: -2px;
}

.toggle-icon {
  color: #6b7280;
  font-size: 15px;
  line-height: 1;
}

.section-fields {
  display: grid;
  gap: 7px;
  padding: 0 8px 8px;
}

.model-field {
  display: grid;
  gap: 3px;
  color: #6b7280;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.model-field textarea {
  width: 100%;
  min-height: 36px;
  padding: 6px 7px;
  resize: vertical;
  color: #374151;
  background: #ffffff;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  font: inherit;
  font-size: 11px;
  font-weight: 400;
  letter-spacing: normal;
  line-height: 1.35;
  text-transform: none;
}
</style>
