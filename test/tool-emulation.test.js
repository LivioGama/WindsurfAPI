import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ToolCallStreamParser, parseToolCallsFromText } from '../src/handlers/tool-emulation.js';

describe('ToolCallStreamParser', () => {
  it('parses XML-format tool calls', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      'Here is the result:\n<tool_call>{"name":"Read","arguments":{"path":"./file.js"}}</tool_call>\nDone.'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 1);
    assert.equal(allCalls[0].name, 'Read');
    assert.ok(JSON.parse(allCalls[0].argumentsJson).path === './file.js');
    assert.ok(r.text.includes('Here is the result:'));
  });

  it('parses <tool_call> without closing tag (SWE-1.x style)', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      'I will list both.<tool_call>{"name":"list_directory","arguments":{"path":"a"}}<tool_call>{"name":"list_directory","arguments":{"path":"b"}}'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 2, `expected 2 tool calls, got ${allCalls.length}`);
    assert.equal(allCalls[0].name, 'list_directory');
    assert.equal(JSON.parse(allCalls[0].argumentsJson).path, 'a');
    assert.equal(allCalls[1].name, 'list_directory');
    assert.equal(JSON.parse(allCalls[1].argumentsJson).path, 'b');
    assert.ok(!r.text.includes('<tool_call>'), `leaked tool_call in text: ${r.text}`);
  });

  it('strips orphan Cascade XML artifacts from text deltas', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      '<tool_call>{"name":"Read","arguments":{"path":"a"}}</tool_call>I\'ll help you.</arg_value></arg_value>'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 1);
    assert.ok(!r.text.includes('</arg_value>'), `leaked arg_value: ${r.text}`);
    assert.ok(r.text.includes("I'll help you."), r.text);
  });

  it('parses <tool_call>NAME{args} (SWE-1.x bare-name format)', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      'I will glob.<tool_call>Glob{"pattern":"*.py"}<tool_call>Glob{"pattern":"**/benchmark*"}'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 2);
    assert.equal(allCalls[0].name, 'Glob');
    assert.equal(JSON.parse(allCalls[0].argumentsJson).pattern, '*.py');
    assert.equal(allCalls[1].name, 'Glob');
    assert.ok(!r.text.includes('<tool_call>'), `leaked: ${r.text}`);
  });

  it('parses <tool_call>NAME attr="value" (attribute-style args)', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      'Let me check.<tool_call>read filePath="./README.md"<tool_call>read filePath="./package.json"<tool_call>read filePath="./src/index.js"'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 3, `got ${allCalls.length}`);
    assert.equal(allCalls[0].name, 'read');
    assert.equal(JSON.parse(allCalls[0].argumentsJson).filePath, './README.md');
    assert.equal(JSON.parse(allCalls[2].argumentsJson).filePath, './src/index.js');
    assert.ok(!r.text.includes('<tool_call>'), `leaked: ${r.text}`);
  });

  it('strips orphan <think>/</think> tags', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed("I'll help.</think></think></think>Done.");
    assert.ok(!r.text.includes('</think>'), `leaked: ${r.text}`);
    assert.equal(r.text, "I'll help.Done.");
  });

  it('parses <tool_call> with JSON body then explicit closer', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      '<tool_call>{"name":"Read","arguments":{"path":"x"}}</tool_call> after text'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 1);
    assert.equal(allCalls[0].name, 'Read');
    assert.ok(r.text.includes(' after text'));
  });

  it('parses bare JSON tool calls', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      '{"name":"Write","arguments":{"path":"a.txt","content":"hello"}}'
    );
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 1);
    assert.equal(allCalls[0].name, 'Write');
  });

  it('handles tool call split across chunks', () => {
    const parser = new ToolCallStreamParser();
    const r1 = parser.feed('<tool_call>{"name":"Rea');
    const r2 = parser.feed('d","arguments":{"path":"x"}}</tool_call>');
    const r3 = parser.flush();
    const allCalls = [...r1.toolCalls, ...r2.toolCalls, ...r3.toolCalls];
    assert.equal(allCalls.length, 1);
    assert.equal(allCalls[0].name, 'Read');
  });

  it('emits text before and after tool calls', () => {
    const parser = new ToolCallStreamParser();
    const r = parser.feed(
      'Before\n<tool_call>{"name":"X","arguments":{}}</tool_call>\nAfter'
    );
    const flush = parser.flush();
    const text = r.text + flush.text;
    assert.ok(text.includes('Before'));
    assert.ok(text.includes('After'));
    assert.ok(!text.includes('<tool_call>'));
  });

  it('handles multiple tool calls in one chunk', () => {
    const parser = new ToolCallStreamParser();
    const input = '<tool_call>{"name":"A","arguments":{}}</tool_call>text<tool_call>{"name":"B","arguments":{}}</tool_call>';
    const r = parser.feed(input);
    const flush = parser.flush();
    const allCalls = [...r.toolCalls, ...flush.toolCalls];
    assert.equal(allCalls.length, 2);
  });
});

describe('parseToolCallsFromText', () => {
  it('extracts tool calls and strips them from text', () => {
    const input = 'Hello\n<tool_call>{"name":"Read","arguments":{"path":"x.js"}}</tool_call>\nWorld';
    const { text, toolCalls } = parseToolCallsFromText(input);
    assert.equal(toolCalls.length, 1);
    assert.equal(toolCalls[0].name, 'Read');
    assert.ok(!text.includes('<tool_call>'));
    assert.ok(text.includes('Hello'));
  });

  it('returns empty array when no tool calls', () => {
    const { text, toolCalls } = parseToolCallsFromText('Just normal text');
    assert.equal(toolCalls.length, 0);
    assert.equal(text, 'Just normal text');
  });
});
