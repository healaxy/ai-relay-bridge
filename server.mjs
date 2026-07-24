// AI relay bridge -- the persistent piece of the real AI calling agent that Supabase Edge Functions
// can't host (their WebSocket lifetime caps out at 150s/400s; a phone call can run longer). Twilio's
// ConversationRelay connects here as a WebSocket client for the duration of one call. It has already
// done speech-to-text and will do text-to-speech on our behalf (that's the HIPAA-eligible piece, once
// the account has a signed BAA) -- this service only ever sees/sends text, never raw audio.
//
// Protocol reference: https://www.twilio.com/docs/voice/conversationrelay/websocket-messages
//   Twilio -> us:  {type:'setup', callSid, customParameters:{...}}
//                  {type:'prompt', voicePrompt, lang, last}
//                  {type:'interrupt', utteranceUntilInterrupt, durationUntilInterruptMs}
//                  {type:'error', description}
//   us -> Twilio:  {type:'text', token, last}          -- spoken via TTS
//                  {type:'end', handoffData}            -- ends the session, hangs up

import { WebSocketServer } from 'ws';
import { createServer } from 'node:http';
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';

const PORT = process.env.PORT || 8080;
// TEST_MODE keeps call content generic/non-clinical until a Twilio BAA is signed (Twilio Security/
// Enterprise Edition required for ConversationRelay to be used with real PHI) -- see the compliance
// note this was built under. Flip to 'false' only once that's in place.
const TEST_MODE = (process.env.AI_CALL_TEST_MODE ?? 'true') !== 'false';

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

const LANGUAGE_LABELS = { en: 'English', es: 'Spanish', hi: 'Hindi' };

const GREETINGS = {
  en: {
    sos: (name) => `Hello, this is Follow-up's automated care line. We received an SOS alert from ${name}. Is now an okay time to talk?`,
    message: (name) => `Hello, this is Follow-up's automated care line. We received your message to your care team, ${name}. Is now an okay time to talk?`,
  },
  es: {
    sos: (name) => `Hola, soy la línea de atención automática de Follow-up. Recibimos una alerta de SOS de ${name}. ¿Es un buen momento para hablar?`,
    message: (name) => `Hola, soy la línea de atención automática de Follow-up. Recibimos su mensaje para su equipo de atención, ${name}. ¿Es un buen momento para hablar?`,
  },
  hi: {
    sos: (name) => `नमस्ते, मैं Follow-up की स्वचालित केयर लाइन बोल रहा/रही हूं। हमें ${name} की ओर से एक SOS अलर्ट मिला है। क्या अभी बात करने का सही समय है?`,
    message: (name) => `नमस्ते, मैं Follow-up की स्वचालित केयर लाइन बोल रहा/रही हूं। हमें ${name} का आपकी केयर टीम को भेजा संदेश मिला है। क्या अभी बात करने का सही समय है?`,
  },
};

function systemPrompt({ trigger, language, patientName, triggerNote }) {
  const langLabel = LANGUAGE_LABELS[language] ?? 'English';
  const base = trigger === 'sos'
    ? `You are an automated healthcare triage phone agent for "Follow-up". You are calling ${patientName} because they triggered an SOS alert. Ask what's happening, assess severity (breathing trouble, pain radiating to arm/jaw, chest tightness), and if anything sounds serious, tell them you're alerting their care team immediately and a provider will call within minutes. Stay calm, warm, and concise -- this is a phone call, so keep each turn to 1-3 short sentences.`
    : `You are an automated healthcare triage phone agent for "Follow-up". You are calling ${patientName} to follow up on a message they sent their care team${triggerNote ? `: "${triggerNote}"` : ''}. Ask what's going on, gauge severity, and either reassure/log it for routine follow-up or escalate if it sounds urgent. Keep each turn to 1-3 short sentences -- this is a phone call.`;
  const language_instr = `Respond only in ${langLabel}.`;
  const endInstr = `When the conversation has reached a natural close (patient has nothing more to add, or you've delivered your closing line), end your final message with the exact token [END_CALL] on its own -- it will be stripped before being spoken.`;
  const testModeInstr = TEST_MODE
    ? `IMPORTANT: This is a TEST call, not a real patient interaction (no BAA is in place with the telephony/AI vendors yet, so no real clinical content should be discussed). Open by saying this is a test call for the Follow-up automated care line demo, and keep the rest of the conversation to a light, generic mock walkthrough rather than real symptoms.`
    : '';
  return [base, language_instr, endInstr, testModeInstr].filter(Boolean).join('\n\n');
}

async function persistSession(sessionId, patch) {
  const { data: existing } = await supabase.from('app_state').select('data').eq('kind', 'ai_call').eq('id', sessionId).maybeSingle();
  const merged = { ...(existing?.data ?? {}), ...patch };
  await supabase.from('app_state').upsert({ kind: 'ai_call', id: sessionId, data: merged });
  return merged;
}

async function summarize(transcript, ctx) {
  try {
    const resp = await anthropic.messages.create({
      model: 'claude-sonnet-4-5',
      max_tokens: 200,
      system: 'Summarize this care-line phone call transcript in 2-3 sentences for a clinician reviewing it later. Note any escalation.',
      messages: [{ role: 'user', content: JSON.stringify(transcript) }],
    });
    return resp.content?.[0]?.text ?? '';
  } catch (err) {
    console.error('Summary generation failed', err);
    return `AI call to ${ctx.patientName} (${ctx.trigger}). Summary unavailable.`;
  }
}

const server = createServer((req, res) => {
  if (req.url === '/health' || req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('ok');
    return;
  }
  res.writeHead(404);
  res.end();
});

const wss = new WebSocketServer({ server, path: '/relay' });

wss.on('connection', (ws) => {
  let ctx = null; // { sessionId, patientId, patientName, trigger, triggerNote, language, callSid }
  let claudeHistory = []; // [{role:'user'|'assistant', content:string}]
  let transcript = []; // [{speaker:'agent'|'patient', text}]

  ws.on('message', async (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    if (msg.type === 'setup') {
      const p = msg.customParameters ?? {};
      ctx = {
        sessionId: p.sessionId ?? `ai-call-unknown-${Date.now()}`,
        patientId: p.patientId ?? '',
        patientName: p.patientName ?? 'the patient',
        trigger: p.trigger ?? 'message',
        triggerNote: p.triggerNote ?? '',
        language: p.language && LANGUAGE_LABELS[p.language] ? p.language : 'en',
        callSid: msg.callSid,
      };
      console.log('setup', ctx);

      const greetingFn = (GREETINGS[ctx.language] ?? GREETINGS.en)[ctx.trigger === 'sos' ? 'sos' : 'message'];
      const greeting = greetingFn(ctx.patientName);
      transcript.push({ speaker: 'agent', text: greeting });
      claudeHistory.push({ role: 'assistant', content: greeting });
      ws.send(JSON.stringify({ type: 'text', token: greeting, last: true }));
      await persistSession(ctx.sessionId, { transcript, status: 'in-progress' });
      return;
    }

    if (msg.type === 'prompt' && msg.last && ctx) {
      const patientText = msg.voicePrompt ?? '';
      transcript.push({ speaker: 'patient', text: patientText });
      claudeHistory.push({ role: 'user', content: patientText });
      await persistSession(ctx.sessionId, { transcript });

      try {
        const resp = await anthropic.messages.create({
          model: 'claude-sonnet-4-5',
          max_tokens: 300,
          system: systemPrompt(ctx),
          messages: claudeHistory,
        });
        let reply = resp.content?.[0]?.text ?? "I'm sorry, could you repeat that?";
        const shouldEnd = reply.includes('[END_CALL]');
        reply = reply.replace('[END_CALL]', '').trim();

        claudeHistory.push({ role: 'assistant', content: reply });
        transcript.push({ speaker: 'agent', text: reply });
        const urgent = ctx.trigger === 'sos' || /\b(escalat|emergency|call.*(right away|immediately)|911)\b/i.test(reply);
        await persistSession(ctx.sessionId, { transcript, urgent });

        ws.send(JSON.stringify({ type: 'text', token: reply, last: true }));

        if (shouldEnd) {
          const summary = await summarize(transcript, ctx);
          await persistSession(ctx.sessionId, { transcript, summary, status: 'completed', endedAt: new Date().toISOString() });
          ws.send(JSON.stringify({ type: 'end', handoffData: JSON.stringify({ reason: 'call-complete' }) }));
        }
      } catch (err) {
        console.error('Claude turn failed', err);
        ws.send(JSON.stringify({ type: 'text', token: "I'm having trouble right now -- your care team will follow up with you directly.", last: true }));
      }
      return;
    }

    if (msg.type === 'interrupt') {
      console.log('interrupt', msg);
      return;
    }

    if (msg.type === 'error') {
      console.error('ConversationRelay error', msg.description);
      return;
    }
  });

  ws.on('close', async () => {
    if (!ctx) return;
    try {
      const summary = await summarize(transcript, ctx);
      await persistSession(ctx.sessionId, { transcript, summary, status: 'completed', endedAt: new Date().toISOString() });
    } catch (err) {
      console.error('Failed to finalize session on close', err);
    }
  });
});

server.listen(PORT, () => {
  console.log(`ai-relay-bridge listening on :${PORT} (TEST_MODE=${TEST_MODE})`);
});
