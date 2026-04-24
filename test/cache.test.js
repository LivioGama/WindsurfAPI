import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { cacheKey, cacheGet, cacheSet, cacheClear } from '../src/cache.js';

beforeEach(() => cacheClear());

describe('cacheKey', () => {
  it('produces deterministic keys', () => {
    const body = { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] };
    assert.equal(cacheKey(body), cacheKey(body));
  });

  it('differs for different models', () => {
    const a = { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] };
    const b = { model: 'claude-4.5-sonnet', messages: [{ role: 'user', content: 'hi' }] };
    assert.notEqual(cacheKey(a), cacheKey(b));
  });

  it('ignores stream flag', () => {
    const a = { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }], stream: true };
    const b = { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }], stream: false };
    assert.equal(cacheKey(a), cacheKey(b));
  });

  it('strips base64 image data from key', () => {
    const withImage = {
      model: 'gpt-4o',
      messages: [{ role: 'user', content: [
        { type: 'text', text: 'describe this' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0KGgo' + 'A'.repeat(10000) } },
      ]}],
    };
    const withDifferentImage = {
      model: 'gpt-4o',
      messages: [{ role: 'user', content: [
        { type: 'text', text: 'describe this' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,DIFFERENT' + 'B'.repeat(10000) } },
      ]}],
    };
    assert.equal(cacheKey(withImage), cacheKey(withDifferentImage));
  });
});

describe('cacheGet / cacheSet', () => {
  it('returns null on miss', () => {
    assert.equal(cacheGet('nonexistent'), null);
  });

  it('stores and retrieves values', () => {
    const value = { text: 'hello', thinking: null };
    cacheSet('key1', value);
    const got = cacheGet('key1');
    assert.deepEqual(got, value);
  });

  it('does not cache empty values', () => {
    cacheSet('empty', null);
    assert.equal(cacheGet('empty'), null);
    cacheSet('empty2', { text: '', chunks: [] });
    assert.equal(cacheGet('empty2'), null);
  });
});
