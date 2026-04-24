import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { resolveModel, getModelInfo, getModelKeysByEnum, MODEL_TIER_ACCESS } from '../src/models.js';

describe('resolveModel', () => {
  it('resolves exact model names', () => {
    assert.equal(resolveModel('gpt-4o'), 'gpt-4o');
  });

  it('resolves case-insensitive aliases', () => {
    assert.equal(resolveModel('GPT-4O'), 'gpt-4o');
  });

  it('resolves Anthropic dated aliases', () => {
    const result = resolveModel('claude-3-5-sonnet-20240620');
    assert.equal(result, 'claude-3.5-sonnet');
  });

  it('resolves Cursor-friendly aliases without claude prefix', () => {
    const result = resolveModel('opus-4.6');
    assert.equal(result, 'claude-opus-4.6');
  });

  it('returns input unchanged for unknown models', () => {
    assert.equal(resolveModel('nonexistent-model-xyz'), 'nonexistent-model-xyz');
  });

  it('returns null for null/empty input', () => {
    assert.equal(resolveModel(null), null);
    assert.equal(resolveModel(''), null);
  });
});

describe('getModelInfo', () => {
  it('returns model info for known model', () => {
    const info = getModelInfo('gpt-4o');
    assert.ok(info);
    assert.ok(info.enumValue > 0 || info.modelUid);
  });

  it('returns null for unknown model', () => {
    assert.equal(getModelInfo('fake-model'), null);
  });
});

describe('getModelKeysByEnum', () => {
  it('returns keys for known enum', () => {
    const info = getModelInfo('gpt-4o');
    if (info?.enumValue) {
      const keys = getModelKeysByEnum(info.enumValue);
      assert.ok(keys.includes('gpt-4o'));
    }
  });

  it('returns empty array for unknown enum', () => {
    assert.deepEqual(getModelKeysByEnum(999999), []);
  });
});

describe('MODEL_TIER_ACCESS', () => {
  it('pro tier includes all models', () => {
    assert.ok(MODEL_TIER_ACCESS.pro.length > 100);
  });

  it('free tier is a small subset', () => {
    assert.ok(MODEL_TIER_ACCESS.free.length <= 5);
    assert.ok(MODEL_TIER_ACCESS.free.includes('gemini-2.5-flash'));
  });

  it('expired tier is empty', () => {
    assert.deepEqual(MODEL_TIER_ACCESS.expired, []);
  });
});
