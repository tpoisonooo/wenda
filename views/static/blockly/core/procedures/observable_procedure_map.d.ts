/**
 * @license
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { IProcedureModel } from '../interfaces/i_procedure_model.js';
import type { Workspace } from '../workspace.js';
import { IProcedureMap } from '../interfaces/i_procedure_map.js';
export declare class ObservableProcedureMap extends Map<string, IProcedureModel> implements IProcedureMap {
    private readonly workspace;
    constructor(workspace: Workspace);
    /**
     * Adds the given procedure model to the procedure map.
     */
    set(id: string, proc: IProcedureModel): this;
    /**
     * Deletes the ProcedureModel with the given ID from the procedure map (if it
     * exists).
     */
    delete(id: string): boolean;
    /**
     * Removes all ProcedureModels from the procedure map.
     */
    clear(): void;
    /**
     * Adds the given ProcedureModel to the map of procedure models, so that
     * blocks can find it.
     */
    add(proc: IProcedureModel): this;
    /**
     * Returns all of the procedures stored in this map.
     */
    getProcedures(): IProcedureModel[];
}
//# sourceMappingURL=observable_procedure_map.d.ts.map