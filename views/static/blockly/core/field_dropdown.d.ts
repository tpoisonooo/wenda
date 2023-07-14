/**
 * @license
 * Copyright 2012 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { FieldConfig, Field } from './field.js';
import { Menu } from './menu.js';
import { MenuItem } from './menuitem.js';
import type { Sentinel } from './utils/sentinel.js';
/**
 * Class for an editable dropdown field.
 *
 * @alias Blockly.FieldDropdown
 */
export declare class FieldDropdown extends Field {
    /** Horizontal distance that a checkmark overhangs the dropdown. */
    static CHECKMARK_OVERHANG: number;
    /**
     * Maximum height of the dropdown menu, as a percentage of the viewport
     * height.
     */
    static MAX_MENU_HEIGHT_VH: number;
    static ARROW_CHAR: string;
    /** A reference to the currently selected menu item. */
    private selectedMenuItem_;
    /** The dropdown menu. */
    protected menu_: Menu | null;
    /**
     * SVG image element if currently selected option is an image, or null.
     */
    private imageElement_;
    /** Tspan based arrow element. */
    private arrow_;
    /** SVG based arrow element. */
    private svgArrow_;
    /**
     * Serializable fields are saved by the serializer, non-serializable fields
     * are not. Editable fields should also be serializable.
     */
    SERIALIZABLE: boolean;
    /** Mouse cursor style when over the hotspot that initiates the editor. */
    CURSOR: string;
    protected menuGenerator_?: MenuGenerator;
    /** A cache of the most recently generated options. */
    private generatedOptions_;
    /**
     * The prefix field label, of common words set after options are trimmed.
     *
     * @internal
     */
    prefixField: string | null;
    /**
     * The suffix field label, of common words set after options are trimmed.
     *
     * @internal
     */
    suffixField: string | null;
    private selectedOption_;
    clickTarget_: SVGElement | null;
    /**
     * @param menuGenerator A non-empty array of options for a dropdown list, or a
     *     function which generates these options. Also accepts Field.SKIP_SETUP
     *     if you wish to skip setup (only used by subclasses that want to handle
     *     configuration and setting the field value after their own constructors
     *     have run).
     * @param opt_validator A function that is called to validate changes to the
     *     field's value. Takes in a language-neutral dropdown option & returns a
     *     validated language-neutral dropdown option, or null to abort the
     *     change.
     * @param opt_config A map of options used to configure the field.
     *     See the [field creation documentation]{@link
     * https://developers.google.com/blockly/guides/create-custom-blocks/fields/built-in-fields/dropdown#creation}
     * for a list of properties this parameter supports.
     * @throws {TypeError} If `menuGenerator` options are incorrectly structured.
     */
    constructor(menuGenerator: MenuGenerator, opt_validator?: Function, opt_config?: FieldConfig);
    constructor(menuGenerator: Sentinel);
    /**
     * Sets the field's value based on the given XML element. Should only be
     * called by Blockly.Xml.
     *
     * @param fieldElement The element containing info about the field's state.
     * @internal
     */
    fromXml(fieldElement: Element): void;
    /**
     * Sets the field's value based on the given state.
     *
     * @param state The state to apply to the dropdown field.
     * @internal
     */
    loadState(state: any): void;
    /**
     * Create the block UI for this dropdown.
     *
     * @internal
     */
    initView(): void;
    /**
     * Whether or not the dropdown should add a border rect.
     *
     * @returns True if the dropdown field should add a border rect.
     */
    protected shouldAddBorderRect_(): boolean;
    /** Create a tspan based arrow. */
    protected createTextArrow_(): void;
    /** Create an SVG based arrow. */
    protected createSVGArrow_(): void;
    /**
     * Create a dropdown menu under the text.
     *
     * @param opt_e Optional mouse event that triggered the field to open, or
     *     undefined if triggered programmatically.
     */
    protected showEditor_(opt_e?: MouseEvent): void;
    /** Create the dropdown editor. */
    private dropdownCreate_;
    /**
     * Disposes of events and DOM-references belonging to the dropdown editor.
     */
    private dropdownDispose_;
    /**
     * Handle an action in the dropdown menu.
     *
     * @param menuItem The MenuItem selected within menu.
     */
    private handleMenuActionEvent_;
    /**
     * Handle the selection of an item in the dropdown menu.
     *
     * @param menu The Menu component clicked.
     * @param menuItem The MenuItem selected within menu.
     */
    protected onItemSelected_(menu: Menu, menuItem: MenuItem): void;
    /**
     * @returns True if the option list is generated by a function.
     *     Otherwise false.
     */
    isOptionListDynamic(): boolean;
    /**
     * Return a list of the options for this dropdown.
     *
     * @param opt_useCache For dynamic options, whether or not to use the cached
     *     options or to re-generate them.
     * @returns A non-empty array of option tuples:
     *     (human-readable text or image, language-neutral name).
     * @throws {TypeError} If generated options are incorrectly structured.
     */
    getOptions(opt_useCache?: boolean): MenuOption[];
    /**
     * Ensure that the input value is a valid language-neutral option.
     *
     * @param opt_newValue The input value.
     * @returns A valid language-neutral option, or null if invalid.
     */
    protected doClassValidation_(opt_newValue?: MenuOption[1]): string | null;
    /**
     * Update the value of this dropdown field.
     *
     * @param newValue The value to be saved. The default validator guarantees
     *     that this is one of the valid dropdown options.
     */
    protected doValueUpdate_(newValue: MenuOption[1]): void;
    /**
     * Updates the dropdown arrow to match the colour/style of the block.
     *
     * @internal
     */
    applyColour(): void;
    /** Draws the border with the correct width. */
    protected render_(): void;
    /**
     * Renders the selected option, which must be an image.
     *
     * @param imageJson Selected option that must be an image.
     */
    private renderSelectedImage_;
    /** Renders the selected option, which must be text. */
    private renderSelectedText_;
    /**
     * Position a drop-down arrow at the appropriate location at render-time.
     *
     * @param x X position the arrow is being rendered at, in px.
     * @param y Y position the arrow is being rendered at, in px.
     * @returns Amount of space the arrow is taking up, in px.
     */
    private positionSVGArrow_;
    /**
     * Use the `getText_` developer hook to override the field's text
     * representation.  Get the selected option text. If the selected option is an
     * image we return the image alt text.
     *
     * @returns Selected option text.
     */
    protected getText_(): string | null;
    /**
     * Construct a FieldDropdown from a JSON arg object.
     *
     * @param options A JSON object with options (options).
     * @returns The new field instance.
     * @nocollapse
     * @internal
     */
    static fromJson(options: FieldDropdownFromJsonConfig): FieldDropdown;
}
/**
 * Definition of a human-readable image dropdown option.
 */
export interface ImageProperties {
    src: string;
    alt: string;
    width: number;
    height: number;
}
/**
 * An individual option in the dropdown menu. The first element is the human-
 * readable value (text or image), and the second element is the language-
 * neutral value.
 */
export declare type MenuOption = [string | ImageProperties, string];
/**
 * A function that generates an array of menu options for FieldDropdown
 * or its descendants.
 */
export declare type MenuGeneratorFunction = (this: FieldDropdown) => MenuOption[];
/**
 * Either an array of menu options or a function that generates an array of
 * menu options for FieldDropdown or its descendants.
 */
export declare type MenuGenerator = MenuOption[] | MenuGeneratorFunction;
/**
 * fromJson config for the dropdown field.
 */
export interface FieldDropdownFromJsonConfig extends FieldConfig {
    options?: MenuOption[];
}
//# sourceMappingURL=field_dropdown.d.ts.map