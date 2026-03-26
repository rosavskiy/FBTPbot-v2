import React, { useState, useRef, useEffect, useCallback } from 'react'
import { api, ChatResponse, SuggestedTopic } from '../api/client'
import ReactMarkdown from 'react-markdown'
import './ChatWidget.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  youtubeLinks?: string[]
  needsEscalation?: boolean
  suggestedTopics?: SuggestedTopic[]
  responseType?: 'answer' | 'clarification'
}

export function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const [activeLlmLabel, setActiveLlmLabel] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Здравствуйте! Чем могу помочь? 💊',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [unread, setUnread] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    if (isOpen) {
      scrollToBottom()
      setUnread(0)
    }
  }, [messages, isOpen, scrollToBottom])

  const sendMessage = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    setMessages(prev => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    setLoading(true)

    try {
      const response: ChatResponse = await api.sendMessage(trimmed, sessionId || undefined)

      if (!sessionId) setSessionId(response.session_id)
      setActiveLlmLabel(response.show_llm_in_chat ? response.llm_label || null : null)

      const botMsg: Message = {
        role: 'assistant',
        content: response.answer,
        youtubeLinks: response.youtube_links,
        needsEscalation: response.needs_escalation,
        suggestedTopics: response.suggested_topics || undefined,
        responseType: response.response_type,
      }

      setMessages(prev => [...prev, botMsg])

      if (!isOpen) setUnread(prev => prev + 1)

      // Автоэскалация при низкой уверенности (не для уточняющих вопросов)
      if (response.needs_escalation && response.session_id && response.response_type !== 'clarification') {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content:
              '💡 Ответ не найден. Свяжитесь с оператором или задайте вопрос иначе.',
          },
        ])
      }
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Ошибка соединения. Попробуйте позже.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleEscalation = async () => {
    if (!sessionId) return
    try {
      const res = await api.createEscalation(sessionId)
      setMessages(prev => [...prev, { role: 'assistant', content: `✅ ${res.message}` }])
    } catch {
      // ignore
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleTopicSelect = async (topicIndex: number) => {
    if (loading) return
    const text = String(topicIndex + 1)
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const response: ChatResponse = await api.sendMessage(text, sessionId || undefined)
      if (!sessionId) setSessionId(response.session_id)
      setActiveLlmLabel(response.show_llm_in_chat ? response.llm_label || null : null)

      const botMsg: Message = {
        role: 'assistant',
        content: response.answer,
        youtubeLinks: response.youtube_links,
        needsEscalation: response.needs_escalation,
        suggestedTopics: response.suggested_topics || undefined,
        responseType: response.response_type,
      }
      setMessages(prev => [...prev, botMsg])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Ошибка соединения. Попробуйте позже.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  // Скрываем виджет на странице /operator
  if (typeof window !== 'undefined' && window.location.pathname.startsWith('/operator')) {
    return null
  }

  return (
    <>
      {/* Кнопка виджета */}
      <button
        className={`widget-toggle ${isOpen ? 'widget-toggle-open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Открыть чат техподдержки"
      >
        {isOpen ? '✕' : '💬'}
        {!isOpen && unread > 0 && <span className="widget-badge">{unread}</span>}
      </button>

      {/* Окно чата */}
      {isOpen && (
        <div className="widget-window">
          <div className="widget-header">
            <div className="widget-header-left">
              <div className="widget-header-icon">Ф</div>
              <div className="widget-header-meta">
                <span>Техподдержка Фармбазис</span>
                {activeLlmLabel && <span className="widget-llm-badge">{activeLlmLabel}</span>}
              </div>
            </div>
            <button onClick={() => setIsOpen(false)} aria-label="Закрыть">✕</button>
          </div>

          <div className="widget-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`widget-msg widget-msg-${msg.role}`}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                {msg.youtubeLinks?.map((link, j) => (
                  <a key={j} href={link} target="_blank" rel="noopener noreferrer" className="widget-yt-link">
                    📹 Видео-инструкция
                  </a>
                ))}
                {/* Кнопки выбора темы (уточнение) */}
                {msg.responseType === 'clarification' && msg.suggestedTopics && msg.suggestedTopics.length > 0 && (
                  <div className="widget-clarification">
                    {msg.suggestedTopics.map((topic, j) => (
                      <button
                        key={j}
                        className="widget-topic-btn"
                        onClick={() => handleTopicSelect(j)}
                        disabled={loading}
                        title={topic.snippet}
                      >
                        {j + 1}. {topic.title}
                      </button>
                    ))}
                    <button
                      className="widget-topic-btn widget-topic-btn--other"
                      onClick={() => {
                        const input = document.querySelector<HTMLInputElement>('.widget-input input')
                        if (input) {
                          input.focus()
                          input.placeholder = 'Опишите проблему подробнее...'
                        }
                      }}
                    >
                      🔍 Не в списке
                    </button>
                  </div>
                )}
                {msg.needsEscalation && msg.responseType !== 'clarification' && (
                  <button className="widget-escalation-btn" onClick={handleEscalation}>
                    📞 Связаться с оператором
                  </button>
                )}
              </div>
            ))}
            {loading && (
              <div className="widget-msg widget-msg-assistant widget-typing">
                <span></span><span></span><span></span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="widget-input">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Задайте вопрос..."
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={!input.trim() || loading}>
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  )
}
