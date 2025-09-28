import { type ChatUIMessage } from '@/components/chat/types'
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs,
  streamText,
} from 'ai'
import { DEFAULT_MODEL } from '@/ai/constants'
import { NextResponse } from 'next/server'
import { getAvailableModels, getModelOptions } from '@/ai/gateway'
import { checkBotId } from 'botid/server'
import { tools } from '@/ai/tools'
import prompt from './prompt.md'

interface BodyData {
  messages: ChatUIMessage[]
  modelId?: string
  reasoningEffort?: 'low' | 'medium'
}

export async function fetchWhiteCircle(
  {
    externalId,
    role,
    content
  }: {
    externalId?: string
    role: 'user' | 'assistant'
    content: string
  }) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 10_000)
  const url = `https://${process.env.WHITECIRCLE_API_REGION}.whitecircle.ai/api/protect/check`

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'content-type': 'application/json',
        'whitecircle-version': '2025-06-15',
        Authorization: `Bearer ${process.env.WHITECIRCLE_API_KEY}`
      },
      body: JSON.stringify({
        external_id: externalId,
        include_context: true, // loads context based on external_id
        include_policy_names: true, // returns names of all the checked policies
        double_check: true, // uses larger model to check for false positives
        messages: [{ content, role }]
      }),
      signal: controller.signal
    })

    clearTimeout(timeout)

    if (!res.ok) {
      return false
    }

    const body = await res.json()
    return Boolean(body?.violation)
  } catch (err) {
    console.error('Error communicating with White Circle', err)
    return false
  }
}

export async function POST(req: Request) {
  const checkResult = await checkBotId()
  if (checkResult.isBot) {
    return NextResponse.json({ error: `Bot detected` }, { status: 403 })
  }

  const [models, { messages, modelId = DEFAULT_MODEL, reasoningEffort }] =
    await Promise.all([getAvailableModels(), req.json() as Promise<BodyData>])

  const model = models.find((model) => model.id === modelId)
  if (!model) {
    return NextResponse.json(
      { error: `Model ${modelId} not found.` },
      { status: 400 }
    )
  }

  return createUIMessageStreamResponse({
    stream: createUIMessageStream({
      originalMessages: messages,
      execute: async ({ writer }) => {
        const messageIdWhiteCircle = `whitecircle-${crypto.randomUUID()}`
        let isViolated = false

        if (process.env.WHITECIRCLE_API_KEY) {
          const filteredMessages = messages
            .filter(m => m.role === 'user' || m.role === 'assistant')

          const lastMessage = filteredMessages[filteredMessages.length - 1]
          const firstMessageId = filteredMessages.find(m => m.id)?.id

          writer.write({
            type: 'reasoning-start',
            id: messageIdWhiteCircle
          })

          isViolated = await fetchWhiteCircle({
            externalId: firstMessageId,
            role: lastMessage.role as 'user' | 'assistant',
            content: lastMessage.parts
              .map(part => part.type === 'text' ? part.text : '')
              .join('\n')
          })

          writer.write({
            type: 'reasoning-end',
            id: messageIdWhiteCircle
          })
        }

        if (isViolated) {
          writer.write({
            type: 'text-start',
            id: messageIdWhiteCircle
          })

          writer.write({
            type: 'text-delta',
            delta: `I cannot assist with this request, please contact support if you believe this is an error.`,
            id: messageIdWhiteCircle
          })

          writer.write({
            type: 'text-end',
            id: messageIdWhiteCircle
          })
        } else {
          const result = streamText({
            ...getModelOptions(modelId, { reasoningEffort }),
            system: prompt,
            messages: convertToModelMessages(
              messages.map((message) => {
                message.parts = message.parts.map((part) => {
                  if (part.type === 'data-report-errors') {
                    return {
                      type: 'text',
                      text:
                        `There are errors in the generated code. This is the summary of the errors we have:\n` +
                        `\`\`\`${part.data.summary}\`\`\`\n` +
                        (part.data.paths?.length
                          ? `The following files may contain errors:\n` +
                          `\`\`\`${part.data.paths?.join('\n')}\`\`\`\n`
                          : '') +
                        `Fix the errors reported.`,
                    }
                  }
                  return part
                })
                return message
              })
            ),
            stopWhen: stepCountIs(20),
            tools: tools({ modelId, writer }),
            onError: (error) => {
              console.error('Error communicating with AI')
              console.error(JSON.stringify(error, null, 2))
            },
          })
          await result.consumeStream()
          writer.merge(
            result.toUIMessageStream({
              sendReasoning: true,
              sendStart: false,
              messageMetadata: () => ({
                model: model.name,
              }),
            })
          )
        }
      },
    }),
  })
}
