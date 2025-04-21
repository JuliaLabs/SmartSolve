using Revise
using Dagger
#using DaggerWebDash
#import DaggerWebDash: GanttPlot, LinePlot
#using TimespanLogging
using DataFrames
using Plots
using GraphViz
using LinearAlgebra
using ScopedValues

const IS_SETUP = Ref{Bool}(false)
function setup()
    if IS_SETUP[]
        return
    end
    ml = TimespanLogging.MultiEventLog()
    ml[:core] = TimespanLogging.Events.CoreMetrics()
    ml[:id] = TimespanLogging.Events.IDMetrics()
    ml[:wsat] = Dagger.Events.WorkerSaturation()
    ml[:loadavg] = TimespanLogging.Events.CPULoadAverages()
    ml[:bytes] = Dagger.Events.BytesAllocd()
    ml[:mem] = TimespanLogging.Events.MemoryFree()
    ml[:esat] = TimespanLogging.Events.EventSaturation()
    ml[:psat] = Dagger.Events.ProcessorSaturation()
    lw = TimespanLogging.Events.LogWindow(20*10^9, :core)
    logs_df = DataFrame([key=>[] for key in keys(ml.consumers)]...)
    ts = DaggerWebDash.TableStorage(logs_df)
    push!(lw.creation_handlers, ts)
    d3r = DaggerWebDash.D3Renderer(8080; seek_store=ts)
    push!(lw.creation_handlers, d3r)
    push!(lw.deletion_handlers, d3r)
    push!(d3r, GanttPlot(:core, :id, :esat, :psat; title="Overview"))
    # TODO: push!(d3r, ProfileViewer(:core, :profile, "Profile Viewer"))
    #push!(d3r, LinePlot(:core, :wsat, "Worker Saturation", "Running Tasks"))
    #push!(d3r, LinePlot(:core, :loadavg, "CPU Load Average", "Average Running Threads"))
    #push!(d3r, LinePlot(:core, :bytes, "Allocated Bytes", "Bytes"))
    #push!(d3r, LinePlot(:core, :mem, "Available Memory", "% Free"))
    #push!(d3r, GraphPlot(:core, :id, :timeline, :profile, "DAG"))
    ml.aggregators[:d3r] = d3r
    ml.aggregators[:logwindow] = lw
    Dagger.Sch.eager_context().log_sink = ml
    IS_SETUP[] = true
end

function generate(A, B, C; nt=Threads.nthreads())
    @info "Generating for $nt threads"
    Dagger.enable_logging!(;metrics=false, all_task_deps=true)
    GC.enable(false)
    Dagger.with_options(;scope=Dagger.scope(;threads=1:nt)) do
        @with Dagger.DATADEPS_SCHEDULE_REUSABLE => false begin
            @time mul!(C, A, B)
        end
    end
    GC.enable(true)
    logs = Dagger.fetch_logs!()
    Dagger.disable_logging!()
    display(Dagger.render_logs(logs, :plots_gantt; target=:execution, color_init_hash=UInt(2)))
    return
end
function generate_all()
    #setup()
    Dagger.MemPool.MEM_RESERVED[] = 0
    A = rand(Blocks(512, 512), 2048, 2048)
    B = rand(Blocks(512, 512), 2048, 2048)
    C = zeros(Blocks(512, 512), 2048, 2048)
    wait.(A.chunks)
    wait.(B.chunks)
    wait.(C.chunks)
    generate(A, B, C, nt=1)
    generate(A, B, C, nt=4)
    generate(A, B, C, nt=8)
    generate(A, B, C, nt=16)
end
generate_all()