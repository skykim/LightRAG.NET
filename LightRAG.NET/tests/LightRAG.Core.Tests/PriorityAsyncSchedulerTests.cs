using System.Collections.Concurrent;
using LightRAG.Core.Concurrency;

namespace LightRAG.Core.Tests;

public class PriorityAsyncSchedulerTests
{
    [Fact]
    public async Task Never_exceeds_max_concurrency()
    {
        var scheduler = new PriorityAsyncScheduler(maxConcurrency: 3);
        var current = 0;
        var peak = 0;
        var sync = new object();

        async Task Work(CancellationToken ct)
        {
            lock (sync) { current++; peak = Math.Max(peak, current); }
            await Task.Delay(20, ct);
            lock (sync) { current--; }
        }

        var tasks = Enumerable.Range(0, 30).Select(_ => scheduler.RunAsync(Work)).ToArray();
        await Task.WhenAll(tasks);

        Assert.True(peak <= 3, $"peak concurrency was {peak}, expected <= 3");
    }

    [Fact]
    public async Task Runs_queued_work_in_priority_order()
    {
        // One slot, occupied first so everything else queues; then drain in priority order.
        var scheduler = new PriorityAsyncScheduler(maxConcurrency: 1);
        var order = new ConcurrentQueue<int>();

        // Occupy the only slot and hold it until the queue is populated.
        var gate = new TaskCompletionSource();
        var blocker = scheduler.RunAsync(async _ => { await gate.Task; });

        // Enqueue work with mixed priorities while the slot is held.
        var lowPrio = scheduler.RunAsync(_ => { order.Enqueue(10); return Task.CompletedTask; }, priority: 10);
        var highPrio = scheduler.RunAsync(_ => { order.Enqueue(1); return Task.CompletedTask; }, priority: 1);
        var midPrio = scheduler.RunAsync(_ => { order.Enqueue(5); return Task.CompletedTask; }, priority: 5);

        // Give the enqueues a moment to register before releasing.
        await Task.Delay(30);
        gate.SetResult();
        await Task.WhenAll(blocker, lowPrio, highPrio, midPrio);

        Assert.Equal([1, 5, 10], order.ToArray());
    }

    [Fact]
    public async Task Times_out_long_running_work()
    {
        var scheduler = new PriorityAsyncScheduler(maxConcurrency: 1);
        await Assert.ThrowsAsync<TimeoutException>(async () =>
            await scheduler.RunAsync(
                async ct => { await Task.Delay(5000, ct); return 0; },
                timeout: TimeSpan.FromMilliseconds(50)));
    }

    [Fact]
    public async Task Releases_slot_after_exception()
    {
        var scheduler = new PriorityAsyncScheduler(maxConcurrency: 1);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await scheduler.RunAsync<int>(_ => throw new InvalidOperationException("boom")));

        // The slot must be free again for subsequent work.
        var result = await scheduler.RunAsync(_ => Task.FromResult(42));
        Assert.Equal(42, result);
    }
}
