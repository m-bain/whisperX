'use client';

import { motion } from 'framer-motion';

export type ViewMode = 'grid' | 'list' | 'kanban';

interface ViewSwitcherProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
}

const VIEWS: { id: ViewMode; label: string; icon: string }[] = [
  { id: 'grid', label: 'Griglia', icon: '⊞' },
  { id: 'list', label: 'Lista', icon: '☰' },
  { id: 'kanban', label: 'Kanban', icon: '⚏' },
];

export function ViewSwitcher({ currentView, onViewChange }: ViewSwitcherProps) {
  return (
    <div className="flex items-center gap-2 bg-gray-800/50 border border-gray-700/50 rounded-xl p-1">
      {VIEWS.map((view) => (
        <motion.button
          key={view.id}
          onClick={() => onViewChange(view.id)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            currentView === view.id
              ? 'text-emerald-300'
              : 'text-gray-400 hover:text-gray-300'
          }`}
        >
          {currentView === view.id && (
            <motion.div
              layoutId="activeView"
              className="absolute inset-0 bg-emerald-500/20 border border-emerald-500/30 rounded-lg"
              transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative flex items-center gap-2">
            <span className="text-lg">{view.icon}</span>
            <span className="hidden sm:inline">{view.label}</span>
          </span>
        </motion.button>
      ))}
    </div>
  );
}
