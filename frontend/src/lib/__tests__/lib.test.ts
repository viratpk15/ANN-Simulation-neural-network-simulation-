
describe('LLM provider status rendering contract', () => {
  const chain = ['groq', 'openrouter', 'ollama']

  it('describes the default fallback chain in order', () => {
    expect(chain.join(' → ')).toBe('groq → openrouter → ollama')
  })

  it('carries every field the AI Diagnosis panel renders', () => {
    const status: LlmStatus = {
      enabled: true,
      chain,
      chain_labels: ['Groq', 'OpenRouter', 'Ollama (local)'],
      providers: [
        { name: 'groq', label: 'Groq', state: 'available', model: 'llama-3.3-70b-versatile' },
        { name: 'openrouter', label: 'OpenRouter', state: 'not_configured', model: 'x', detail: 'API key missing' },
        { name: 'ollama', label: 'Ollama (local)', state: 'available', model: 'llama3.1' },
      ],
      models: { groq: 'llama-3.3-70b-versatile', openrouter: 'x', ollama: 'llama3.1' },
      timeout_s: 30,
      max_retries: 1,
      note: 'ok',
    }
    expect(status.providers.map((p) => p.name)).toEqual(chain)
    expect(status.providers[1].state).toBe('not_configured')
    expect(status.providers[1].detail).toBe('API key missing')
    // no provider field may hold a key
    expect(Object.values(status.models).every((m) => !/key/i.test(m))).toBe(true)
  })

  it('surfaces the fallback outcome and message', () => {
    const info: LlmInfo = {
      enabled: true,
      available: true,
      explanation: 'text',
      provider: 'openrouter',
      label: 'OpenRouter',
      model: 'meta-llama/llama-3.3-70b-instruct',
      outcome: 'fallback',
      message: 'Explanation generated using OpenRouter fallback.',
      fallback_used: true,
      chain,
    }
    expect(info.fallback_used).toBe(true)
    expect(info.message).toContain('OpenRouter fallback')
    expect(info.label).toBe('OpenRouter')
  })

  it('marks total failure without breaking the deterministic findings', () => {
    const info: LlmInfo = {
      enabled: true,
      available: false,
      explanation: null,
      outcome: 'unavailable',
      message: 'LLM explanation unavailable. Deterministic diagnosis is still available.',
      error: 'All configured LLM providers failed (...).',
      chain,
    }
    expect(info.available).toBe(false)
    expect(info.explanation).toBeNull()
    expect(info.message).toContain('Deterministic diagnosis is still available')
  })

  it('defines a dot style for every provider state', () => {
    const states: ProviderState[] = [
      'available', 'not_configured', 'active', 'failed', 'skipped', 'disabled',
    ]
    expect(new Set(states).size).toBe(6)
  })
})

import { describe, expect, it } from 'vitest'
import { activationColor, fmt, histogram, pct, weightColor } from '../format'
import { shapeText, specFrom, withDefaults } from '../../store/useAppStore'
import type { LayerKind, LlmInfo, LlmStatus, ProviderState } from '../../types'

describe('format helpers', () => {
  it('formats numbers sanely', () => {
    expect(fmt(1.234567)).toBe('1.2346')
    expect(fmt(0.0000001)).toMatch(/e/i)
    expect(fmt(null)).toBe('—')
    expect(pct(0.914)).toBe('91.4%')
  })

  it('histograms count every value once', () => {
    const values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    const h = histogram(values, 5)
    expect(h.reduce((a, b) => a + b.count, 0)).toBe(10)
    expect(h).toHaveLength(5)
  })

  it('produces strong colours for extreme weights', () => {
    expect(weightColor(2, -2, 2)).toContain('rgba(248')
    expect(weightColor(-2, -2, 2)).toContain('rgba(96')
    expect(activationColor(1, 0, 1)).not.toBe(activationColor(0, 0, 1))
  })
})

describe('shapeText', () => {
  it('renders a vector width on its own', () => {
    expect(shapeText([784])).toBe('784')
    expect(shapeText([16])).toBe('16')
  })

  it('renders image shapes channels-last as H × W × C', () => {
    expect(shapeText([28, 28, 1])).toBe('28 × 28 × 1')
    expect(shapeText([8, 8, 3])).toBe('8 × 8 × 3')
  })

  it('handles missing shapes', () => {
    expect(shapeText(null)).toBe('—')
    expect(shapeText(undefined)).toBe('—')
    expect(shapeText([])).toBe('—')
  })
})

describe('withDefaults', () => {
  it('fills in every new parameter for a legacy v1 layer', () => {
    // exactly what a project saved before the new layers existed contains
    const p = withDefaults('dense', { neurons: 8, activation: 'tanh', use_bias: true, init: 'xavier' })
    expect(p.neurons).toBe(8)
    expect(p.activation).toBe('tanh')
    // new fields take their defaults rather than being undefined
    expect(p.momentum).toBe(0.1)
    expect(p.eps).toBe(1e-5)
    expect(p.affine).toBe(true)
    expect(p.dropout_mode).toBe('train_only')
    expect(p.input_shape ?? null).toBe(null)
  })

  it('preserves values that were explicitly saved', () => {
    const p = withDefaults('conv2d', { filters: 32, kernel_size: 5, padding_mode: 'same' })
    expect(p.filters).toBe(32)
    expect(p.kernel_size).toBe(5)
    expect(p.padding_mode).toBe('same')
  })

  it('keeps a saved image shape on the input layer', () => {
    const p = withDefaults('input', { features: 784, input_shape: [28, 28, 1] })
    expect(p.input_shape).toEqual([28, 28, 1])
    expect(p.features).toBe(784)
  })

  it('gives every supported kind a usable default', () => {
    const kinds: LayerKind[] = ['input', 'dense', 'activation', 'dropout', 'batchnorm',
      'flatten', 'conv2d', 'maxpool', 'avgpool', 'globalavgpool', 'output']
    for (const k of kinds) {
      const p = withDefaults(k)
      expect(p).toBeTruthy()
      expect(typeof p.activation).toBe('string')
    }
    expect(withDefaults('maxpool').pool_size).toBe(2)
    expect(withDefaults('conv2d').kernel_size).toBe(3)
    expect(withDefaults('batchnorm').affine).toBe(true)
  })

  it('falls back when a required value is null', () => {
    expect(withDefaults('output', { neurons: null }).neurons).toBe(2)
    expect(withDefaults('input', { features: null }).features).toBe(2)
  })
})

describe('specFrom (canvas graph → network spec)', () => {
  it('maps reactflow nodes/edges into the backend spec format', () => {
    const nodes = [
      { id: 'input', type: 'layerNode', position: { x: 0, y: 0 }, data: { kind: 'input', label: 'Input', params: withDefaults('input', { features: 2 }) } },
      { id: 'out', type: 'layerNode', position: { x: 200, y: 0 }, data: { kind: 'output', label: 'Out', params: withDefaults('output', { neurons: 2, activation: 'softmax' }) } },
    ] as never
    const edges = [{ id: 'e1', source: 'input', target: 'out' }] as never
    const spec = specFrom(nodes, edges, 'P', { swishy: 'x*sigmoid(x)' })
    expect(spec.name).toBe('P')
    expect(spec.layers).toHaveLength(2)
    expect(spec.layers[1].params.activation).toBe('softmax')
    expect(spec.edges[0]).toEqual({ id: 'e1', source: 'input', target: 'out' })
    expect(spec.custom_activations.swishy).toBe('x*sigmoid(x)')
  })

  it('carries the new layer kinds and their parameters into the spec', () => {
    const nodes = [
      { id: 'i', type: 'layerNode', position: { x: 0, y: 0 }, data: { kind: 'input', label: 'I', params: withDefaults('input', { features: 64, input_shape: [8, 8, 1] }) } },
      { id: 'c', type: 'layerNode', position: { x: 1, y: 0 }, data: { kind: 'conv2d', label: 'C', params: withDefaults('conv2d', { filters: 8, kernel_size: 3 }) } },
      { id: 'b', type: 'layerNode', position: { x: 2, y: 0 }, data: { kind: 'batchnorm', label: 'B', params: withDefaults('batchnorm') } },
      { id: 'a', type: 'layerNode', position: { x: 3, y: 0 }, data: { kind: 'activation', label: 'A', params: withDefaults('activation', { activation: 'gelu' }) } },
      { id: 'p', type: 'layerNode', position: { x: 4, y: 0 }, data: { kind: 'maxpool', label: 'P', params: withDefaults('maxpool', { pool_size: 2, pool_stride: 2 }) } },
      { id: 'g', type: 'layerNode', position: { x: 5, y: 0 }, data: { kind: 'globalavgpool', label: 'G', params: withDefaults('globalavgpool') } },
    ] as never
    const spec = specFrom(nodes, [] as never, 'CNN', {})
    expect(spec.layers.map((l) => l.kind)).toEqual(
      ['input', 'conv2d', 'batchnorm', 'activation', 'maxpool', 'globalavgpool'])
    expect(spec.layers[0].params.input_shape).toEqual([8, 8, 1])
    expect(spec.layers[1].params.filters).toBe(8)
    expect(spec.layers[2].params.momentum).toBe(0.1)
    expect(spec.layers[3].params.activation).toBe('gelu')
    expect(spec.layers[4].params.pool_size).toBe(2)
  })
})
