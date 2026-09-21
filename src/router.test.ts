import { expect, test } from "bun:test"
import { poolNormalized } from "./router"

test("poolNormalized averages the weighted tokens and unit-normalizes", () => {
    // 3 tokens x 2 dims; only tokens 0 and 2 are pooled (weight 1/2 each).
    const proj = new Float32Array([3, 0, 99, 99, 1, 0])
    const weights = new Float32Array([0.5, 0, 0.5])

    const out = poolNormalized({ weights, proj, count: 3, projDim: 2 })

    expect([...out]).toEqual([1, 0]) // mean is [2, 0] -> normalized [1, 0]
})

test("poolNormalized survives an all-zero lane", () => {
    const out = poolNormalized({
        weights: new Float32Array(3),
        proj: new Float32Array(6),
        count: 3,
        projDim: 2,
    })

    expect([...out].every(Number.isFinite)).toBe(true)
})
