import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { sanitizeText, PathSanitizeStream, sanitizeToolCall } from '../src/sanitize.js';

describe('sanitizeText', () => {
  it('rewrites /tmp/windsurf-workspace paths to relative', () => {
    assert.equal(sanitizeText('/tmp/windsurf-workspace/src/index.js'), './src/index.js');
  });

  it('rewrites bare /tmp/windsurf-workspace to .', () => {
    assert.equal(sanitizeText('/tmp/windsurf-workspace'), '.');
  });

  it('rewrites per-account workspace paths', () => {
    assert.equal(
      sanitizeText('/home/user/projects/workspace-abc12345/package.json'),
      './package.json'
    );
  });

  it('rewrites /opt/windsurf to [internal]', () => {
    assert.equal(sanitizeText('/opt/windsurf/language_server'), '[internal]');
  });

  it('leaves normal text unchanged', () => {
    const text = 'Hello, this is a normal response.';
    assert.equal(sanitizeText(text), text);
  });

  it('handles multiple patterns in one string', () => {
    const input = 'Editing /tmp/windsurf-workspace/a.js and /opt/windsurf/bin';
    const result = sanitizeText(input);
    assert.equal(result, 'Editing ./a.js and [internal]');
  });

  it('returns non-strings unchanged', () => {
    assert.equal(sanitizeText(null), null);
    assert.equal(sanitizeText(undefined), undefined);
    assert.equal(sanitizeText(''), '');
  });
});

describe('PathSanitizeStream', () => {
  it('sanitizes a complete path in one chunk', () => {
    const stream = new PathSanitizeStream();
    const out = stream.feed('/tmp/windsurf-workspace/file.js is here');
    const rest = stream.flush();
    assert.equal(out + rest, './file.js is here');
  });

  it('handles path split across chunks', () => {
    const stream = new PathSanitizeStream();
    let result = '';
    result += stream.feed('Look at /tmp/windsurf');
    result += stream.feed('-workspace/config.yaml for details');
    result += stream.flush();
    assert.equal(result, 'Look at ./config.yaml for details');
  });

  it('handles partial prefix at buffer end', () => {
    const stream = new PathSanitizeStream();
    let result = '';
    result += stream.feed('path is /tmp/win');
    result += stream.feed('dsurf-workspace/x.js done');
    result += stream.flush();
    assert.equal(result, 'path is ./x.js done');
  });

  it('flushes clean text immediately', () => {
    const stream = new PathSanitizeStream();
    const out = stream.feed('Hello world ');
    assert.equal(out, 'Hello world ');
  });
});

describe('sanitizeToolCall', () => {
  it('sanitizes argumentsJson paths', () => {
    const tc = { name: 'Read', argumentsJson: '{"path":"/tmp/windsurf-workspace/f.js"}' };
    const result = sanitizeToolCall(tc);
    assert.equal(result.argumentsJson, '{"path":"./f.js"}');
  });

  it('sanitizes input object string values', () => {
    const tc = { name: 'Read', input: { file_path: '/home/user/projects/workspace-abc12345/src/x.ts' } };
    const result = sanitizeToolCall(tc);
    assert.equal(result.input.file_path, './src/x.ts');
  });

  it('returns null/undefined unchanged', () => {
    assert.equal(sanitizeToolCall(null), null);
    assert.equal(sanitizeToolCall(undefined), undefined);
  });
});
