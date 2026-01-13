'use client';

interface CostTrackerProps {
  totalCost: number;
  totalMinutes: number;
}

export function CostTracker({ totalCost, totalMinutes }: CostTrackerProps) {
  // Budget settings (can be configurable)
  const monthlyBudget = 50; // €50/month
  const budgetPercentage = Math.min((totalCost / monthlyBudget) * 100, 100);
  const isOverBudget = totalCost > monthlyBudget;

  // Calculate projected cost (based on daily usage)
  const today = new Date();
  const daysInMonth = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
  const currentDay = today.getDate();
  const projectedCost = (totalCost / currentDay) * daysInMonth;

  // Circle progress calculation
  const circumference = 2 * Math.PI * 70; // radius = 70
  const progress = (budgetPercentage / 100) * circumference;

  return (
    <div className="bg-gradient-to-br from-slate-900/80 to-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
      <h3 className="text-lg font-semibold text-slate-100 mb-6">
        Tracking Costi
      </h3>

      {/* Circular Progress */}
      <div className="relative w-48 h-48 mx-auto mb-6">
        <svg className="transform -rotate-90 w-48 h-48">
          {/* Background circle */}
          <circle
            cx="96"
            cy="96"
            r="70"
            stroke="currentColor"
            strokeWidth="12"
            fill="none"
            className="text-slate-800"
          />
          {/* Progress circle */}
          <circle
            cx="96"
            cy="96"
            r="70"
            stroke="currentColor"
            strokeWidth="12"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            className={`transition-all duration-1000 ${
              isOverBudget ? 'text-red-500' : 'text-violet-500'
            }`}
            strokeLinecap="round"
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <p className="text-4xl font-bold text-slate-100">
            {budgetPercentage.toFixed(0)}%
          </p>
          <p className="text-xs text-slate-400 mt-1">del budget</p>
        </div>
      </div>

      {/* Cost Details */}
      <div className="space-y-3">
        <div className="flex items-center justify-between py-3 border-b border-slate-800/50">
          <span className="text-sm text-slate-400">Speso questo mese</span>
          <span className="text-sm font-semibold text-slate-100">
            €{totalCost.toFixed(2)}
          </span>
        </div>

        <div className="flex items-center justify-between py-3 border-b border-slate-800/50">
          <span className="text-sm text-slate-400">Budget mensile</span>
          <span className="text-sm font-semibold text-slate-100">
            €{monthlyBudget.toFixed(2)}
          </span>
        </div>

        <div className="flex items-center justify-between py-3 border-b border-slate-800/50">
          <span className="text-sm text-slate-400">Costo per minuto</span>
          <span className="text-sm font-semibold text-slate-100">
            €{totalMinutes > 0 ? (totalCost / totalMinutes).toFixed(3) : '0.000'}
          </span>
        </div>

        <div className="flex items-center justify-between py-3">
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
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <p className="text-xs text-red-400">
              ⚠️ Hai superato il budget mensile di €{(totalCost - monthlyBudget).toFixed(2)}
            </p>
          </div>
        </div>
      )}

      {/* Savings Tip */}
      {!isOverBudget && projectedCost < monthlyBudget * 0.5 && (
        <div className="mt-4 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-xs text-emerald-400">
              ✓ Ottimo! Stai risparmiando circa €{(monthlyBudget - projectedCost).toFixed(2)} questo mese
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
