'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { UserMenu } from '@/components/UserMenu';

export default function HomePage() {
  const options = [
    {
      id: 1,
      title: 'Option 1: Minimalist Professional',
      description: 'Interfaccia pulita e veloce. Perfetta per MVP e power users.',
      href: '/dashboard-1',
      icon: '⚡',
      color: 'from-blue-500 to-cyan-500',
      features: [
        'Setup veloce',
        'Tabella sortable/filterable',
        'Performance ottimali',
        'Design professionale',
      ],
      difficulty: 'Facile',
    },
    {
      id: 2,
      title: 'Option 2: Visual Analytics',
      description: 'Dashboard ricca di grafici e insights. Perfetta per business.',
      href: '/dashboard-2',
      icon: '📊',
      color: 'from-violet-500 to-fuchsia-500',
      features: [
        'Grafici dettagliati',
        'Tracking budget',
        'Analytics avanzate',
        'Look premium',
      ],
      difficulty: 'Media',
    },
    {
      id: 3,
      title: 'Option 3: Card-based Modern',
      description: 'UI moderna con animazioni. Perfetta per massimo engagement.',
      href: '/dashboard-3',
      icon: '🎴',
      color: 'from-emerald-500 to-teal-500',
      features: [
        'Animazioni fluide',
        'Grid/List/Kanban',
        'Drag & drop',
        'Design trendy',
      ],
      difficulty: 'Media',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          {/* Top bar with user menu */}
          <div className="flex justify-end mb-4">
            <UserMenu />
          </div>
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 mb-4">
              WhisperX Dashboard
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Scegli la dashboard che preferisci e testala subito.
              Tutte hanno le stesse funzionalità, cambia solo lo stile!
            </p>
          </motion.div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-16">
        {/* Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-12 p-6 bg-gradient-to-r from-blue-500/10 to-violet-500/10 border border-blue-500/20 rounded-xl"
        >
          <h2 className="text-lg font-semibold text-white mb-2">
            ✅ Tutte le funzionalità richieste sono implementate:
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">🎵</span>
              <div>
                <p className="font-medium text-gray-200">Formati Audio</p>
                <p className="text-sm text-gray-400">MP3, M4A, WAV, FLAC, OGG, WebM</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">⏱️</span>
              <div>
                <p className="font-medium text-gray-200">Timer Real-time</p>
                <p className="text-sm text-gray-400">Stato con timer durante elaborazione</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">📄</span>
              <div>
                <p className="font-medium text-gray-200">Export Multiplo</p>
                <p className="text-sm text-gray-400">TXT, SRT, VTT, JSON</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Dashboard Options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {options.map((option, index) => (
            <motion.div
              key={option.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * (index + 1) }}
              className="group"
            >
              <Link href={option.href}>
                <div className="relative h-full">
                  {/* Glow effect */}
                  <div className={`absolute -inset-0.5 bg-gradient-to-r ${option.color} rounded-2xl opacity-0 group-hover:opacity-20 blur transition-opacity duration-300`} />

                  {/* Card */}
                  <div className="relative h-full bg-gray-900/50 backdrop-blur-sm border border-gray-800 rounded-xl p-8 hover:border-gray-700 transition-all duration-300">
                    {/* Icon */}
                    <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${option.color} flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform duration-300`}>
                      {option.icon}
                    </div>

                    {/* Title */}
                    <h3 className="text-2xl font-bold text-white mb-3 group-hover:bg-clip-text group-hover:text-transparent group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-gray-400 transition-all">
                      {option.title}
                    </h3>

                    {/* Description */}
                    <p className="text-gray-400 mb-6">
                      {option.description}
                    </p>

                    {/* Features */}
                    <div className="space-y-2 mb-6">
                      {option.features.map((feature, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm text-gray-300">
                          <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          {feature}
                        </div>
                      ))}
                    </div>

                    {/* Difficulty Badge */}
                    <div className="flex items-center justify-between pt-6 border-t border-gray-800">
                      <span className="text-sm text-gray-500">
                        Difficoltà: <span className="text-gray-300 font-medium">{option.difficulty}</span>
                      </span>
                      <div className={`px-3 py-1 bg-gradient-to-r ${option.color} rounded-lg text-white text-sm font-medium`}>
                        Prova →
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Bottom Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-16 text-center"
        >
          <p className="text-gray-500 mb-4">
            💡 Tip: Puoi testare tutte e 3 e poi scegliere la tua preferita!
          </p>
          <div className="flex items-center justify-center gap-4 text-sm text-gray-600">
            <span>🔥 Hot reload attivo</span>
            <span>•</span>
            <span>🎨 Tailwind CSS</span>
            <span>•</span>
            <span>⚡ Next.js 14</span>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
