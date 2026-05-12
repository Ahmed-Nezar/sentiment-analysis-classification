export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    const targets = {
      api1: env.BGE_M3_API_URL || "https://8000-dep-API1/predict",
      api2: env.QWEN_API_URL || "https://8000-dep-API2/predict",
    };

    const api = url.searchParams.get("api") || "api1";
    const target = targets[api];

    if (!target) {
      return new Response(
        JSON.stringify({ error: "Invalid API target" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const allowedOrigins = [
      "https://ahmed-nezar.github.io",
      "http://localhost:5173",
      "http://127.0.0.1:5173",
    ];

    const origin = request.headers.get("Origin");

    const corsHeaders = {
      "Access-Control-Allow-Origin": allowedOrigins.includes(origin)
        ? origin
        : "null",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders,
      });
    }

    if (request.method !== "POST") {
      return new Response(
        JSON.stringify({ error: "Method not allowed" }),
        {
          status: 405,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        }
      );
    }

    const response = await fetch(target, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${env.TEXT_CLASSIFICATION_API_KEY}`,
      },
      body: await request.text(),
    });

    return new Response(await response.text(), {
      status: response.status,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    });
  },
};
