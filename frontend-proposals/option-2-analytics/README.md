# Option 2: Visual Analytics Dashboard

Dashboard ricca di dati e visualizzazioni ispirata a Descript e Otter.ai.

## 🎨 Design Filosofia

"Data is beautiful" - Focus su insights attraverso visualizzazioni interattive. Perfetta per business users e chi vuole analytics dettagliate.

## ✨ Features Uniche

### 📊 Analytics Avanzate

- **Usage Chart**: Grafico a linee dei minuti trascritti per giorno/settimana/mese
- **Cost Tracker**: Tracking spese con proiezioni e budget
- **Language Distribution**: Pie chart delle lingue più usate
- **Speaker Analytics**: Statistiche sui speaker riconosciuti
- **Timeline View**: Vista temporale delle trascrizioni
- **Heatmap**: Utilizzo per ora del giorno

### 🎯 Tutte le funzionalità base di Option 1, più:

- **Dashboard widgets personalizzabili**
- **Export con preview in real-time**
- **Filtri avanzati** (data range, lingua, speaker count)
- **Statistiche comparative** (vs. settimana scorsa, vs. mese scorso)
- **Notifiche in-app** per completamenti
- **Multi-select operations** (bulk delete, bulk export)

## 📁 Componenti Principali

### Dashboard.tsx - Layout a 3 colonne

```tsx
'use client';

import { UsageChart } from './UsageChart';
import { CostTracker } from './CostTracker';
import { LanguageDistribution } from './LanguageDistribution';
import { TimelineView } from './TimelineView';
import { StatsCards } from './StatsCards';
import { RecentTranscriptions } from './RecentTranscriptions';

export function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header con search globale */}
      <Header />

      <div className="max-w-[1600px] mx-auto px-6 py-8">
        {/* Stats Cards Grid */}
        <StatsCards />

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-12 gap-6 mt-8">
          {/* Left Column - Charts */}
          <div className="col-span-8 space-y-6">
            <UsageChart /> {/* Line chart - minuti per giorno */}
            <TimelineView /> {/* Timeline delle trascrizioni */}
          </div>

          {/* Right Column - Summary & Actions */}
          <div className="col-span-4 space-y-6">
            <CostTracker /> {/* Spese totali + proiezioni */}
            <LanguageDistribution /> {/* Pie chart lingue */}
            <RecentTranscriptions /> {/* Lista recenti */}
          </div>
        </div>

        {/* Bottom Section - Detailed Table */}
        <div className="mt-8">
          <AdvancedTranscriptionsTable />
        </div>
      </div>
    </div>
  );
}
```

### UsageChart.tsx - Grafico utilizzo

```tsx
'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useState } from 'react';

type TimeRange = '7d' | '30d' | '90d';

export function UsageChart() {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');

  // Fetch usage data per timeRange
  const data = useUsageData(timeRange);

  return (
    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">
            Utilizzo nel Tempo
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            Minuti trascritti per giorno
          </p>
        </div>

        {/* Time Range Selector */}
        <div className="flex items-center gap-2 bg-slate-800/50 p-1 rounded-lg">
          {['7d', '30d', '90d'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range as TimeRange)}
              className={`px-3 py-1.5 text-sm rounded-md transition-all ${
                timeRange === range
                  ? 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {range === '7d' ? '7 giorni' : range === '30d' ? '30 giorni' : '90 giorni'}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <defs>
              <linearGradient id="colorMinutes" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="date"
              stroke="#64748b"
              tick={{ fill: '#94a3b8' }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fill: '#94a3b8' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '8px'
              }}
            />
            <Line
              type="monotone"
              dataKey="minutes"
              stroke="#8b5cf6"
              strokeWidth={3}
              dot={{ fill: '#8b5cf6', r: 4 }}
              activeDot={{ r: 6 }}
              fill="url(#colorMinutes)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-slate-800/50">
        <div>
          <p className="text-xs text-slate-400">Media Giornaliera</p>
          <p className="text-lg font-semibold text-slate-100 mt-1">
            {calculateAverage(data)} min
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Picco Massimo</p>
          <p className="text-lg font-semibold text-slate-100 mt-1">
            {Math.max(...data.map(d => d.minutes))} min
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Totale Periodo</p>
          <p className="text-lg font-semibold text-slate-100 mt-1">
            {data.reduce((sum, d) => sum + d.minutes, 0)} min
          </p>
        </div>
      </div>
    </div>
  );
}
```

### CostTracker.tsx - Tracking costi

```tsx
'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

export function CostTracker() {
  const { totalCost, monthlyBudget, projectedCost } = useCostData();

  const budgetPercentage = (totalCost / monthlyBudget) * 100;
  const isOverBudget = budgetPercentage > 100;

  return (
    <div className="bg-gradient-to-br from-slate-900/80 to-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
      <h3 className="text-lg font-semibold text-slate-100 mb-4">
        Tracking Costi
      </h3>

      {/* Budget Progress Circle */}
      <div className="relative w-40 h-40 mx-auto">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={[
                { value: totalCost },
                { value: Math.max(0, monthlyBudget - totalCost) }
              ]}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={70}
              startAngle={90}
              endAngle={-270}
              dataKey="value"
            >
              <Cell fill={isOverBudget ? '#ef4444' : '#8b5cf6'} />
              <Cell fill="#334155" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <p className="text-3xl font-bold text-slate-100">
            {budgetPercentage.toFixed(0)}%
          </p>
          <p className="text-xs text-slate-400">del budget</p>
        </div>
      </div>

      {/* Cost Details */}
      <div className="mt-6 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Speso questo mese</span>
          <span className="text-sm font-semibold text-slate-100">
            €{totalCost.toFixed(2)}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Budget mensile</span>
          <span className="text-sm font-semibold text-slate-100">
            €{monthlyBudget.toFixed(2)}
          </span>
        </div>

        <div className="flex items-center justify-between pt-3 border-t border-slate-800/50">
          <span className="text-sm text-slate-400">Proiezione fine mese</span>
          <span className={`text-sm font-semibold ${
            projectedCost > monthlyBudget ? 'text-orange-400' : 'text-emerald-400'
          }`}>
            €{projectedCost.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Warning */}
      {isOverBudget && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          <p className="text-xs text-red-400">
            ⚠️ Hai superato il budget mensile
          </p>
        </div>
      )}
    </div>
  );
}
```

## 🚀 Setup

Stessi requisiti di Option 1, più:

```bash
npm install recharts date-fns
```

## 🎨 Color Scheme

```css
/* Gradients */
--gradient-primary: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%);
--gradient-secondary: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
--gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);

/* Background */
--bg-primary: #020617;   /* slate-950 */
--bg-secondary: #0f172a; /* slate-900 */
--bg-tertiary: #1e293b;  /* slate-800 */

/* Text */
--text-primary: #f1f5f9;   /* slate-100 */
--text-secondary: #94a3b8; /* slate-400 */
```

## 📊 Metrics Dashboard

Questa opzione è perfetta se vuoi:

- **Analytics dettagliate** del tuo utilizzo
- **Visualizzazioni** chiare e intuitive
- **Proiezioni e trends**
- **Export reports** PDF (con grafici)
- **Dashboard esecutiva** per presentazioni

## 🎯 Pro & Contro

### ✅ Pro

- **Ricca di insights**: Tantissimi dati utili
- **Visivamente attraente**: Grafici colorati
- **Business-ready**: Perfetta per report
- **Decisional**: Aiuta a prendere decisioni informate
- **Professional**: Look premium e curato

### ❌ Contro

- **Più complessa**: Richiede tempo per capire
- **Performance**: Più pesante (grafici)
- **Overwhelming**: Può essere troppo per utenti casual
- **Learning curve**: Non immediata da usare

## 📸 Features Preview

- **Usage Chart**: Trend utilizzo temporale
- **Cost Tracker**: Budget e proiezioni
- **Language Distribution**: Pie chart lingue
- **Timeline**: Vista cronologica
- **Heatmap**: Utilizzo orario
- **Advanced Filters**: Filtri multipli combinabili
- **Export Reports**: PDF con charts inclusi

## 🔗 Recommended For

- **Business users**: Che necessitano di reporting
- **Team accounts**: Con budget da tracciare
- **Power users**: Che amano i dati
- **Analysts**: Che vogliono insights dettagliati
