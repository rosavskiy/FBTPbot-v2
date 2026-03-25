import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ChatPage } from './chat/ChatPage'
import { OperatorPanel } from './operator/OperatorPanel'
import { ChatWidget } from './widget/ChatWidget'
import './styles/global.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/operator" element={<OperatorPanel />} />
      </Routes>
      {/* Виджет чата доступен на всех страницах */}
      <ChatWidget />
    </BrowserRouter>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
