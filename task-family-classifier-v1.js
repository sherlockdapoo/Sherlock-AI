/* Shared Sherlock AI task-family classifier. Keep the UI validator and neural worker on this exact implementation. */
(function (root) {
  "use strict";
  root.sherlockTaskFamilyIndex = function sherlockTaskFamilyIndex(prompt) {
    const value = String(prompt || "").toLowerCase();
    if (/\b(?:plus|minus|multiplied|divide|divided|times|add)\b/.test(value)) return 0;
    if (/%|\bpercent(?:age)?\b/.test(value)) return 1;
    if (/\bconvert\b|\bare equal to\b|\bchange\b.*\binto\b/.test(value)) return 2;
    if (/^(?=.*\d)(?!.*\bexplore books\b).*\b(?:box|shop|bags?|groups?|shared|added|sells?|remain|altogether|marbles|stickers|pencils|books|cards)\b/.test(value)) return 3;
    if (/\bsequence\b/.test(value)) return 4;
    if (/\balphabet\b|\bletter\b|\bascending order\b|\bdescending order\b|\binstruction\b/.test(value)) return 5;
    const numericOperation = /\d+(?:\.\d+)?\s*(?:\+|−|-|×|\*|÷|\/)\s*\d+(?:\.\d+)?/.test(value);
    const verbalOperation = /\b(?:multiply|multiplication|addition|subtract|subtraction|sum|difference|product|quotient)\b/.test(value);
    if (numericOperation || verbalOperation) return 0;
    return 6;
  };
})(self);
