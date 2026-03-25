import React, { useState, useRef, useEffect, useCallback } from 'react'
import { api, ChatResponse, SuggestedTopic } from '../api/client'
import ReactMarkdown from 'react-markdown'
import './ChatPage.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  confidence?: number
  needsEscalation?: boolean
  youtubeLinks?: string[]
  sourceArticles?: string[]
  suggestedTopics?: SuggestedTopic[]
  responseType?: 'answer' | 'clarification'
}

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content:
        'Здравствуйте! Я ИИ-ассистент техподдержки Фармбазис. Задайте вопрос по работе с программой.',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [showEscalation, setShowEscalation] = useState(false)
  const [escalationSent, setEscalationSent] = useState(false)
  const [contactInfo, setContactInfo] = useState('')
  const [escalationReason, setEscalationReason] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const sendMessage = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    const userMessage: Message = { role: 'user', content: trimmed }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response: ChatResponse = await api.sendMessage(
        trimmed,
        sessionId || undefined
      )

      if (!sessionId) {
        setSessionId(response.session_id)
      }

      const botMessage: Message = {
        role: 'assistant',
        content: response.answer,
        confidence: response.confidence,
        needsEscalation: response.needs_escalation,
        youtubeLinks: response.youtube_links,
        sourceArticles: response.source_articles,
        suggestedTopics: response.suggested_topics || undefined,
        responseType: response.response_type,
      }

      setMessages(prev => [...prev, botMessage])

      if (response.needs_escalation && response.response_type !== 'clarification') {
        setShowEscalation(true)
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content:
            'Ошибка обработки запроса. Попробуйте позже или обратитесь к оператору.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleEscalation = async () => {
    if (!sessionId) return

    try {
      const response = await api.createEscalation(
        sessionId,
        escalationReason || undefined,
        contactInfo || undefined
      )

      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `✅ ${response.message}`,
        },
      ])

      setEscalationSent(true)
      setShowEscalation(false)
    } catch {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Не удалось создать заявку. Попробуйте ещё раз.',
        },
      ])
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleFeedback = async (rating: number) => {
    if (!sessionId) return
    try {
      await api.sendFeedback(sessionId, rating, messages.length - 1)
    } catch {
      // silently ignore
    }
  }

  const handleTopicSelect = async (topicIndex: number) => {
    if (loading) return
    const text = String(topicIndex + 1)
    const userMessage: Message = { role: 'user', content: text }
    setMessages(prev => [...prev, userMessage])
    setLoading(true)

    try {
      const response: ChatResponse = await api.sendMessage(text, sessionId || undefined)
      if (!sessionId) setSessionId(response.session_id)

      const botMessage: Message = {
        role: 'assistant',
        content: response.answer,
        confidence: response.confidence,
        needsEscalation: response.needs_escalation,
        youtubeLinks: response.youtube_links,
        sourceArticles: response.source_articles,
        responseType: response.response_type,
        suggestedTopics: response.suggested_topics || undefined,
      }
      setMessages(prev => [...prev, botMessage])

      if (response.needs_escalation && response.response_type !== 'clarification') {
        setShowEscalation(true)
      }
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Ошибка обработки. Попробуйте позже.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="chat-header-logo">
          <div className="chat-header-icon">💊</div>
          <div>
            <h1>Фармбазис — Техподдержка</h1>
            <span className="chat-header-subtitle">ИИ-ассистент</span>
          </div>
        </div>
      </header>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message chat-message-${msg.role}`}>
            <div className="chat-message-avatar">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="chat-message-content">
              <ReactMarkdown>{msg.content}</ReactMarkdown>

              {/* YouTube ссылки */}
              {msg.youtubeLinks && msg.youtubeLinks.length > 0 && (
                <div className="chat-youtube-links">
                  <strong>📹 Видео-инструкции:</strong>
                  {msg.youtubeLinks.map((link, j) => (
                    <a key={j} href={link} target="_blank" rel="noopener noreferrer">
                      {link}
                    </a>
                  ))}
                </div>
              )}

              {/* Кнопки выбора темы (уточнение) */}
              {msg.responseType === 'clarification' && msg.suggestedTopics && msg.suggestedTopics.length > 0 && (
                <div className="chat-clarification">
                  {msg.suggestedTopics.map((topic, j) => (
                    <button
                      key={j}
                      className="chat-topic-btn"
                      onClick={() => handleTopicSelect(j)}
                      disabled={loading}
                      title={topic.snippet}
                    >
                      {j + 1}. {topic.title}
                    </button>
                  ))}
                  <button
                    className="chat-topic-btn chat-topic-btn--other"
                    onClick={() => {
                      const input = document.querySelector<HTMLTextAreaElement>('.chat-input-area textarea')
                      if (input) {
                        input.focus()
                        input.placeholder = 'Опишите проблему подробнее...'
                      }
                    }}
                  >
                    🔍 Моя проблема не в списке
                  </button>
                </div>
              )}

              {/* Feedback */}
              {msg.role === 'assistant' && i > 0 && (
                <div className="chat-feedback">
                  <span>Ответ полезен?</span>
                  <button onClick={() => handleFeedback(5)} title="Полезно">👍</button>
                  <button onClick={() => handleFeedback(1)} title="Не полезно">👎</button>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message chat-message-assistant">
            <div className="chat-message-avatar">🤖</div>
            <div className="chat-message-content chat-typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Блок эскалации */}
      {showEscalation && !escalationSent && (
        <div className="chat-escalation">
          <p>
            🔔 Ответ не найден. Связаться с оператором?
          </p>
          <input
            type="text"
            placeholder="Ваш email или телефон (необязательно)"
            value={contactInfo}
            onChange={e => setContactInfo(e.target.value)}
          />
          <input
            type="text"
            placeholder="Уточните вопрос (необязательно)"
            value={escalationReason}
            onChange={e => setEscalationReason(e.target.value)}
          />
          <div className="chat-escalation-actions">
            <button className="btn-primary" onClick={handleEscalation}>
              Связаться с оператором
            </button>
            <button className="btn-secondary" onClick={() => setShowEscalation(false)}>
              Продолжить с ботом
            </button>
          </div>
        </div>
      )}

      {/* Поле ввода */}
      <div className="chat-input-area">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Задайте вопрос по работе с программой..."
          rows={1}
          disabled={loading}
        />
        <button
          className="chat-send-btn"
          onClick={sendMessage}
          disabled={!input.trim() || loading}
        >
          ➤
        </button>
      </div>
    </div>
  )
}
