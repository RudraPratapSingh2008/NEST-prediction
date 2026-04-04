module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const apiKey = process.env.GROQ_API_KEY || '';
  if (!apiKey) {
    res.status(500).json({ error: 'Missing GROQ_API_KEY environment variable' });
    return;
  }

  // Health check path used by the UI AI Status button.
  if (req.body && req.body.ping) {
    res.status(200).json({ ok: true, provider: 'groq' });
    return;
  }

  const systemPrompt = (req.body && req.body.systemPrompt) || '';
  const userPrompt = (req.body && req.body.userPrompt) || '';

  if (!systemPrompt || !userPrompt) {
    res.status(400).json({ error: 'Missing prompt payload' });
    return;
  }

  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        temperature: 0.2,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        response_format: { type: 'json_object' },
      }),
    });

    if (!groqRes.ok) {
      const errText = await groqRes.text();
      res.status(groqRes.status).json({ error: 'Groq request failed', detail: errText.slice(0, 800) });
      return;
    }

    const data = await groqRes.json();
    const raw = data && data.choices && data.choices[0] && data.choices[0].message
      ? data.choices[0].message.content
      : '{}';

    res.status(200).json({ raw });
  } catch (err) {
    res.status(500).json({ error: 'AI assist server error', detail: String((err && err.message) || err) });
  }
};
