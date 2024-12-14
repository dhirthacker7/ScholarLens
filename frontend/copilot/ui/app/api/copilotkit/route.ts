import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from '@copilotkit/runtime';
import OpenAI from 'openai';
import { NextRequest } from 'next/server';

// Initialize OpenAI client directly (v4.x uses `OpenAI()` instead of `Configuration`)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
});

// Initialize OpenAI Adapter with the OpenAI client
const serviceAdapter = new OpenAIAdapter({ client: openai });

// Create Copilot Runtime with the service adapter
const runtime = new CopilotRuntime({ serviceAdapter });

export const POST = async (req: NextRequest) => {
  try {
    // Setup the request handler using CopilotKit's endpoint utility
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime,
      serviceAdapter,
      endpoint: '/api/copilotkit',
    });

    // Handle the incoming request
    return await handleRequest(req);
  } catch (error) {
    console.error('Error in POST request:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
};
