/**
 * @license
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * The interface for the data model of a procedure parameter.
 *
 * @namespace Blockly.IParameterModel
 */
/**
 * A data model for a procedure.
 */
export interface IParameterModel {
    /**
     * Sets the name of this parameter to the given name.
     */
    setName(name: string): this;
    /**
     * Sets the types of this parameter to the given type.
     */
    setTypes(types: string[]): this;
    /**
     * Returns the name of this parameter.
     */
    getName(): string;
    /**
     * Return the types of this parameter.
     */
    getTypes(): string[];
    /**
     * Returns the unique language-neutral ID for the parameter.
     *
     * This represents the identify of the variable model which does not change
     * over time.
     */
    getId(): string;
}
//# sourceMappingURL=i_parameter_model.d.ts.map