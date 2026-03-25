import React, { useState, useEffect, useCallback } from 'react'
import { api, EscalationDetail } from '../api/client'
import './OperatorPanel.css'

export function OperatorPanel() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('op_token'))
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loginError, setLoginError] = useState('')

  const [escalations, setEscalations] = useState<EscalationDetail[]>([])
  const [pendingCount, setPendingCount] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [replyText, setReplyText] = useState('')
  const [closeTicket, setCloseTicket] = useState(false)
  const [filter, setFilter] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const loadEscalations = useCallback(async () => {
    if (!token) return
    setLoading(true)
    try {
      const data = await api.getEscalations(token, filter || undefined)
      setEscalations(data.escalations)
      setPendingCount(data.pending_count)
    } catch {
      setToken(null)
      localStorage.removeItem('op_token')
    } finally {
      setLoading(false)
    }
  }, [token, filter])

  useEffect(() => {
    loadEscalations()
    const interval = setInterval(loadEscalations, 15000) // Обновление каждые 15с
    return () => clearInterval(interval)
  }, [loadEscalations])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const res = await api.operatorLogin(username, password)
      setToken(res.token)
      localStorage.setItem('op_token', res.token)
      setLoginError('')
    } catch {
      setLoginError('Неверный логин или пароль')
    }
  }

  const handleReply = async () => {
    if (!token || !selectedId || !replyText.trim()) return
    try {
      await api.operatorReply(token, selectedId, replyText, closeTicket)
      setReplyText('')
      setCloseTicket(false)
      loadEscalations()
    } catch {
      alert('Ошибка отправки ответа')
    }
  }

  const handleLogout = () => {
    setToken(null)
    localStorage.removeItem('op_token')
  }

  const selectedEscalation = escalations.find(e => e.escalation_id === selectedId)

  const statusLabel: Record<string, string> = {
    pending: '⏳ Ожидает',
    in_progress: '🔄 В работе',
    resolved: '✅ Решено',
    closed: '🔒 Закрыто',
  }

  const statusClass: Record<string, string> = {
    pending: 'status-pending',
    in_progress: 'status-progress',
    resolved: 'status-resolved',
    closed: 'status-closed',
  }

  // Login form
  if (!token) {
    return (
      <div className="op-login">
        <div className="op-login-card">
          <h1>💊 Фармбазис — Панель ТП</h1>
          <form onSubmit={handleLogin}>
            <input
              type="text"
              placeholder="Логин"
              value={username}
              onChange={e => setUsername(e.target.value)}
            />
            <input
              type="password"
              placeholder="Пароль"
              value={password}
              onChange={e => setPassword(e.target.value)}
            />
            {loginError && <p className="op-error">{loginError}</p>}
            <button type="submit" className="btn-primary">Войти</button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="op-panel">
      {/* Sidebar */}
      <div className="op-sidebar">
        <div className="op-sidebar-header">
          <h2>💊 Техподдержка</h2>
          <div className="op-sidebar-info">
            <span className="op-pending-badge">{pendingCount} ожидают</span>
            <button className="op-logout-btn" onClick={handleLogout}>Выход</button>
          </div>
        </div>

        <div className="op-filters">
          <button
            className={!filter ? 'active' : ''}
            onClick={() => setFilter('')}
          >
            Все
          </button>
          <button
            className={filter === 'pending' ? 'active' : ''}
            onClick={() => setFilter('pending')}
          >
            Ожидают
          </button>
          <button
            className={filter === 'in_progress' ? 'active' : ''}
            onClick={() => setFilter('in_progress')}
          >
            В работе
          </button>
          <button
            className={filter === 'resolved' ? 'active' : ''}
            onClick={() => setFilter('resolved')}
          >
            Решено
          </button>
        </div>

        <div className="op-list">
          {loading && escalations.length === 0 && (
            <p className="op-empty">Загрузка...</p>
          )}
          {!loading && escalations.length === 0 && (
            <p className="op-empty">Нет заявок</p>
          )}
          {escalations.map(esc => (
            <div
              key={esc.escalation_id}
              className={`op-item ${selectedId === esc.escalation_id ? 'op-item-active' : ''}`}
              onClick={() => setSelectedId(esc.escalation_id)}
            >
              <div className="op-item-top">
                <span className={`op-status ${statusClass[esc.status]}`}>
                  {statusLabel[esc.status] || esc.status}
                </span>
                <span className="op-time">
                  {new Date(esc.created_at).toLocaleString('ru')}
                </span>
              </div>
              <p className="op-item-preview">
                {esc.reason || esc.chat_history[esc.chat_history.length - 1]?.content?.slice(0, 80) || 'Нет описания'}
              </p>
              {esc.contact_info && (
                <span className="op-contact">📞 {esc.contact_info}</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Detail view */}
      <div className="op-detail">
        {!selectedEscalation ? (
          <div className="op-detail-empty">
            <p>Выберите заявку из списка слева</p>
          </div>
        ) : (
          <>
            <div className="op-detail-header">
              <div>
                <h3>Заявка #{selectedEscalation.escalation_id.slice(0, 8)}</h3>
                <span className={`op-status ${statusClass[selectedEscalation.status]}`}>
                  {statusLabel[selectedEscalation.status]}
                </span>
              </div>
              {selectedEscalation.contact_info && (
                <span className="op-detail-contact">
                  📞 {selectedEscalation.contact_info}
                </span>
              )}
            </div>

            {selectedEscalation.reason && (
              <div className="op-reason">
                <strong>Причина:</strong> {selectedEscalation.reason}
              </div>
            )}

            <div className="op-chat-history">
              {selectedEscalation.chat_history.map((msg, i) => (
                <div key={i} className={`op-chat-msg op-chat-${msg.role}`}>
                  <span className="op-chat-role">
                    {msg.role === 'user' ? '👤 Пользователь' : '🤖 Бот'}
                  </span>
                  <p>{msg.content}</p>
                </div>
              ))}
            </div>

            {(selectedEscalation.status === 'pending' || selectedEscalation.status === 'in_progress') && (
              <div className="op-reply-area">
                <textarea
                  placeholder="Ваш ответ пользователю..."
                  value={replyText}
                  onChange={e => setReplyText(e.target.value)}
                  rows={3}
                />
                <div className="op-reply-actions">
                  <label>
                    <input
                      type="checkbox"
                      checked={closeTicket}
                      onChange={e => setCloseTicket(e.target.checked)}
                    />
                    Закрыть заявку
                  </label>
                  <button
                    className="btn-primary"
                    onClick={handleReply}
                    disabled={!replyText.trim()}
                  >
                    Отправить ответ
                  </button>
                </div>
              </div>
            )}

            {selectedEscalation.operator_notes && (
              <div className="op-notes">
                <strong>Заметки оператора:</strong>
                <p>{selectedEscalation.operator_notes}</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
