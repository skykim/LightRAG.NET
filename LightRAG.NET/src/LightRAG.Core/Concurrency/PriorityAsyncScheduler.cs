using LightRAG.Core.Configuration;

namespace LightRAG.Core.Concurrency;

/// <summary>
/// Priority-aware concurrency limiter, ported from the essential semantics of
/// <c>priority_limit_async_func_call</c> in <c>lightrag/utils.py</c>.
///
/// All LLM/embedding calls for a given role route through one scheduler instance so that:
/// <list type="bullet">
///   <item>At most <c>maxConcurrency</c> calls run at once.</item>
///   <item>When slots are scarce, queued calls run in <c>(priority, sequence)</c> order —
///         lower priority value first, FIFO within the same priority.</item>
///   <item>Each call is bounded by an optional timeout.</item>
/// </list>
/// The Python implementation's health-check / stuck-task recovery machinery (needed for a
/// long-running multi-process server) is intentionally omitted from this single-process port.
/// A <see cref="SortedSet{T}"/> backs the wait queue (netstandard2.1 has no <c>PriorityQueue</c>).
/// </summary>
public sealed class PriorityAsyncScheduler
{
    private sealed record Waiter(int Priority, long Seq, TaskCompletionSource<bool> Tcs);

    private sealed class WaiterComparer : IComparer<Waiter>
    {
        public static readonly WaiterComparer Instance = new();

        public int Compare(Waiter? x, Waiter? y)
        {
            if (x is null || y is null)
            {
                return Comparer<object?>.Default.Compare(x, y);
            }
            var byPriority = x.Priority.CompareTo(y.Priority);
            return byPriority != 0 ? byPriority : x.Seq.CompareTo(y.Seq);
        }
    }

    private readonly object _gate = new();
    private readonly SortedSet<Waiter> _waiters = new(WaiterComparer.Instance);
    private readonly TimeSpan? _defaultTimeout;
    private int _available;
    private long _seq;

    public PriorityAsyncScheduler(int maxConcurrency, TimeSpan? defaultTimeout = null)
    {
        if (maxConcurrency < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxConcurrency), "Must be at least 1.");
        }
        _available = maxConcurrency;
        _defaultTimeout = defaultTimeout;
        MaxConcurrency = maxConcurrency;
    }

    public int MaxConcurrency { get; }

    /// <summary>Run a function under concurrency + priority control, returning its result.</summary>
    public async Task<T> RunAsync<T>(
        Func<CancellationToken, Task<T>> work,
        int priority = Constants.DefaultProcessingPriority,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        await AcquireAsync(priority, cancellationToken).ConfigureAwait(false);
        try
        {
            var effectiveTimeout = timeout ?? _defaultTimeout;
            if (effectiveTimeout is null)
            {
                return await work(cancellationToken).ConfigureAwait(false);
            }

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(effectiveTimeout.Value);
            try
            {
                return await work(timeoutCts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
            {
                throw new TimeoutException($"Operation timed out after {effectiveTimeout.Value.TotalSeconds:0.#}s.");
            }
        }
        finally
        {
            Release();
        }
    }

    /// <summary>Run a function with no return value under concurrency + priority control.</summary>
    public Task RunAsync(
        Func<CancellationToken, Task> work,
        int priority = Constants.DefaultProcessingPriority,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
        => RunAsync<object?>(async ct => { await work(ct).ConfigureAwait(false); return null; },
            priority, timeout, cancellationToken);

    private Task AcquireAsync(int priority, CancellationToken cancellationToken)
    {
        if (cancellationToken.IsCancellationRequested)
        {
            return Task.FromCanceled(cancellationToken);
        }

        Waiter waiter;
        lock (_gate)
        {
            if (_available > 0)
            {
                _available--;
                return Task.CompletedTask;
            }

            var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            waiter = new Waiter(priority, _seq++, tcs);
            _waiters.Add(waiter);
        }

        // Cancellation while queued: mark the waiter canceled; Release() skips dead waiters.
        if (cancellationToken.CanBeCanceled)
        {
            var registration = cancellationToken.Register(static state => ((Waiter)state!).Tcs.TrySetCanceled(), waiter);
            waiter.Tcs.Task.ContinueWith(static (_, reg) => ((CancellationTokenRegistration)reg!).Dispose(),
                registration, TaskScheduler.Default);
        }

        return waiter.Tcs.Task;
    }

    private void Release()
    {
        lock (_gate)
        {
            // Hand the slot to the highest-priority live waiter; skip ones already canceled.
            while (_waiters.Count > 0)
            {
                var next = _waiters.Min!;
                _waiters.Remove(next);
                if (next.Tcs.TrySetResult(true))
                {
                    return; // slot transferred to this waiter
                }
                // waiter was canceled before acquiring; discard and try the next.
            }
            _available++;
        }
    }
}
