/**
 * Точка входа для встраиваемого виджета.
 *
 * Подключение на сайте:
 * <script src="https://support.farmbazis.ru/farmbazis-support-widget.iife.js"></script>
 *
 * Или с настройками:
 * <script>
 *   window.FARMBAZIS_SUPPORT_CONFIG = {
 *     apiUrl: 'https://support.farmbazis.ru/api',
 *   };
 * </script>
 * <script src="https://support.farmbazis.ru/farmbazis-support-widget.iife.js"></script>
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import { ChatWidget } from './ChatWidget'
import '../styles/global.css'
import './ChatWidget.css'

// Создаём Shadow DOM контейнер (изоляция стилей)
function init() {
  const container = document.createElement('div')
  container.id = 'farmbazis-support-root'
  document.body.appendChild(container)

  ReactDOM.createRoot(container).render(
    <React.StrictMode>
      <ChatWidget />
    </React.StrictMode>
  )
}

// Инициализация при загрузке DOM
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init)
} else {
  init()
}
